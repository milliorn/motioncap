//! motioncap: webcam-based security motion capture. See `CLAUDE.md` and
//! `docs/adr/` for architecture and design-decision context.

/// Rolling pre-buffer of recent frames/audio (see `RingBuffer`).
mod buffer;
/// Camera and audio capture callbacks.
mod capture;
/// CLI argument parsing.
mod config;
/// YOLO object-detection inference.
mod detect;
/// Background-subtraction motion gate.
mod motion;
/// Shared `OpenCV` conversion helpers.
mod opencv_utils;
/// Output file/folder naming.
mod paths;
/// Opt-in live preview window.
mod preview;
/// Recording lifecycle and ffmpeg-backed encoding.
mod recorder;
/// Startup dependency checks.
mod startup;
/// YOLO-detection-to-trigger evaluation.
mod triggers;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};

use buffer::{RingBuffer, TimestampedAudio, TimestampedFrame};
use config::Config;
use detect::Detector;
use motion::MotionGate;
use paths::clip_path;
use preview::PreviewWindow;
use recorder::{RecordingEvent, RecordingEventParams};

/// An event that's been started (ffmpeg spawned) but whose pre-event buffer
/// hasn't been written yet. Kept separate from `RecordingEvent` construction
/// (see `RecordingEvent::start`'s docs) so the detection loop never blocks on
/// writing dozens of pre-buffer frames -- the writer thread seeds it as the
/// first thing it does once it sees a pending event, then it becomes a
/// normal actively-written event.
struct PendingEvent {
    /// The started recording (ffmpeg already spawned) awaiting its pre-buffer seed.
    event: RecordingEvent,
    /// Pre-trigger frames to seed into `event` once the writer thread picks it up.
    pre_frames: Vec<TimestampedFrame>,
    /// Pre-trigger audio to seed into `event` once the writer thread picks it up.
    pre_audio: Vec<TimestampedAudio>,
}

/// Shared state for the currently in-progress recording, if any. Starts as
/// `Pending` (ffmpeg spawned, pre-buffer not yet written) so the writer
/// thread can seed it without blocking whichever thread created it; once
/// seeded it becomes `Active` and receives normal steady-paced writes.
enum ActiveEvent {
    /// No recording is in progress.
    None,
    /// ffmpeg has been spawned but the pre-event buffer hasn't been seeded yet.
    Pending(PendingEvent),
    /// The event is seeded and receiving normal steady-paced writes.
    Active(RecordingEvent),
}

impl ActiveEvent {
    /// Whether any recording (pending or active) is currently in progress.
    const fn is_some(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// The active `RecordingEvent`, if one is seeded and receiving writes.
    const fn as_recording_mut(&mut self) -> Option<&mut RecordingEvent> {
        match self {
            Self::None | Self::Pending(_) => None,
            Self::Active(event) => Some(event),
        }
    }

    /// Takes the in-progress recording (if any), leaving `None` behind.
    fn take(&mut self) -> Option<RecordingEvent> {
        match std::mem::replace(self, Self::None) {
            Self::None => None,
            Self::Pending(pending) => Some(pending.event),
            Self::Active(event) => Some(event),
        }
    }
}

/// Signals when the writer thread has completed its post-shutdown
/// last-chance drain (see `run_recording_writer_loop`), so the detection
/// loop's shutdown path can block on it instead of finalizing the active
/// event prematurely. A condvar rather than a busy-polled flag, since this
/// is a one-shot handshake during shutdown, not a recurring cadence.
#[derive(Default)]
struct WriterDrained {
    /// Set to `true` once the writer thread's final drain has completed.
    done: Mutex<bool>,
    /// Notified when `done` is set, to wake `wait`'s blocked receiver.
    condvar: Condvar,
}

impl WriterDrained {
    /// Marks the final drain as complete and wakes any thread blocked in `wait`.
    fn signal(&self) {
        *self.done.lock().expect("writer-drained lock poisoned") = true;
        self.condvar.notify_one();
    }

    /// Blocks until `signal` has been called.
    fn wait(&self) {
        let guard = self.done.lock().expect("writer-drained lock poisoned");
        drop(
            self.condvar
                .wait_while(guard, |done| !*done)
                .expect("writer-drained lock poisoned"),
        );
    }
}

/// Finalizes whatever recording is in progress (if any) on shutdown.
///
/// Deliberately does *not* use `ActiveEvent::take`, which collapses
/// `Pending`/`Active` down to a bare `RecordingEvent`: a `Pending` event has
/// ffmpeg spawned but no frames/audio written yet -- that only happens once
/// the writer thread picks it up (see `ActiveEvent`'s docs). Finishing it
/// unseeded would hand `finish` an empty video/audio pair, producing a
/// malformed mux instead of a usable (if short) clip, so it must be seeded
/// here first, same as the writer thread would have done.
fn finish_event_on_shutdown(
    active_event: &Mutex<ActiveEvent>,
    writer_drained: &WriterDrained,
) -> Result<()> {
    // The writer thread does its own last-chance drain on shutdown (see
    // `run_recording_writer_loop`) so trailing footage captured while this
    // thread was mid-inference still lands in the clip. Wait for that drain
    // to actually happen before taking/finishing the event -- otherwise this
    // thread can race the writer thread and finish the clip first, in which
    // case the writer's later drain finds `ActiveEvent::None` and silently
    // drops that trailing footage instead of writing it.
    writer_drained.wait();

    let taken = std::mem::replace(
        &mut *active_event.lock().expect("active event lock poisoned"),
        ActiveEvent::None,
    );

    let event = match taken {
        ActiveEvent::None => None,
        ActiveEvent::Pending(mut pending) => {
            if let Err(err) = pending.event.seed(&pending.pre_frames, &pending.pre_audio) {
                log::error!("failed to seed pre-buffer into new recording: {err:?}");
            }
            Some(pending.event)
        }
        ActiveEvent::Active(event) => Some(event),
    };

    if let Some(event) = event {
        event.finish()?;
        log::info!("recording closed on shutdown");
    }

    Ok(())
}

/// Motion-gate + YOLO evaluation cadence. Kept separate from the recording
/// frame rate below since inference cost doesn't scale down usefully at
/// higher polling rates -- 15fps is plenty for deciding whether a subject is
/// still present.
const DETECTION_FRAME_RATE: u32 = 15;
/// Poll interval derived from `DETECTION_FRAME_RATE`.
const DETECTION_POLL_INTERVAL: Duration = Duration::from_millis(1000 / DETECTION_FRAME_RATE as u64);

/// Recorded video frame rate, used by the writer thread and the video
/// encoder. Measured (via traced ring-buffer frame timestamps under real
/// running conditions -- all threads active, real RGB decode load) at ~18fps
/// average delivery for this camera, well short of the 50-65fps seen in
/// isolated capture-only testing. Polling faster than the camera actually
/// delivers just makes the writer re-write stale frames, which plays back as
/// stutter/perceived speed-up (measured ~42% duplicate frame writes at
/// 30fps vs. 0% at 15fps). 15fps is the safe ceiling until the writer tracks
/// per-frame identity to skip real duplicates.
const RECORDING_FRAME_RATE: u32 = 15;
/// Poll interval derived from `RECORDING_FRAME_RATE`.
const RECORDING_POLL_INTERVAL: Duration = Duration::from_millis(1000 / RECORDING_FRAME_RATE as u64);

/// Live preview window refresh rate (diagnostic only; see `preview.rs`).
const PREVIEW_FRAME_RATE: u32 = 30;
/// Poll interval derived from `PREVIEW_FRAME_RATE`.
const PREVIEW_POLL_INTERVAL: Duration = Duration::from_millis(1000 / PREVIEW_FRAME_RATE as u64);

/// Log file name written under `--output-dir` (see `init_logging`).
const LOG_FILE_NAME: &str = "motioncap.log";

/// How long a camera stall (see `FrameLiveness`) must persist before
/// `run_detection_loop` tears down and rebuilds the capture stream, rather
/// than continuing to wait for it to recover on its own.
///
/// Deliberately much longer than `recorder::MAX_FRAME_STALL` (1.5s): that
/// threshold exists to stop feeding stale frames into detection/recording
/// within a couple of seconds, which is far too trigger-happy to also gate
/// tearing down and reopening the OS camera handle -- doing that on every
/// brief stall would thrash the device and could itself induce more stalls.
/// This threshold instead assumes the camera is genuinely gone (see
/// `capture::camera::reconnect_camera_stream`'s doc comment for why nokhwa
/// never recovers from this on its own) and a full stream rebuild is
/// warranted.
const CAMERA_RECONNECT_STALL: Duration = Duration::from_secs(15);

/// Minimum time between reconnect attempts once the camera is believed dead,
/// so a camera that fails to reopen (e.g. genuinely unplugged) doesn't get a
/// reopen attempt on every single detection poll while it's absent.
const CAMERA_RECONNECT_COOLDOWN: Duration = Duration::from_secs(10);

/// Writes every log line to both the given file and stderr, since this
/// process runs long-lived and unattended (per `CLAUDE.md`) -- stderr alone
/// is lost the moment the terminal/session that launched it goes away, but
/// keeping stderr too means interactive/`--preview` runs still see live
/// diagnostics without needing to tail the file.
struct TeeWriter {
    /// The persistent log file under `--output-dir`.
    file: std::fs::File,
}

/// Runs both fallible sink operations unconditionally -- never short-circuit
/// with `?` after just the first -- and propagates the first error, if any.
/// One sink failing (e.g. a full disk, or stderr closed under a supervisor)
/// must never silently suppress the other from being attempted.
fn both(a: std::io::Result<()>, b: std::io::Result<()>) -> std::io::Result<()> {
    a?;
    b?;
    Ok(())
}

impl std::io::Write for TeeWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        both(self.file.write_all(buf), std::io::stderr().write_all(buf))?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        both(self.file.flush(), std::io::stderr().flush())
    }
}

/// Initializes logging to write every line to both `<output_dir>/motioncap.log`
/// and stderr (see `TeeWriter`), honoring `RUST_LOG` exactly as a bare
/// `env_logger::init()` would otherwise. `output_dir` is created if it
/// doesn't exist yet, since this may run before anything else has created it.
fn init_logging(output_dir: &std::path::Path) -> Result<()> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create output directory {}", output_dir.display()))?;

    let log_path = output_dir.join(LOG_FILE_NAME);
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("failed to open log file {}", log_path.display()))?;

    env_logger::Builder::from_env(env_logger::Env::default())
        .target(env_logger::Target::Pipe(Box::new(TeeWriter { file })))
        .init();

    Ok(())
}

/// Starts capture, the detection worker, the recording writer, and (if
/// `--preview` is set) the preview loop, then blocks on the preview loop
/// until shutdown.
fn main() -> Result<()> {
    let config = Config::parse_args();
    init_logging(&config.output_dir)?;

    startup::check_dependencies(&config)?;

    let pre_buffer = Duration::from_secs(u64::from(config.pre_buffer_secs));
    let ring_buffer = Arc::new(Mutex::new(RingBuffer::new(pre_buffer)));

    let camera = capture::camera::start_camera_capture(
        config.camera_device.as_deref(),
        Arc::clone(&ring_buffer),
    )?;
    // Shared with the detection worker so it can rebuild the stream in place
    // when `CAMERA_RECONNECT_STALL` trips (see `FrameLiveness`); held here
    // too so the capture thread stays alive for the process lifetime even
    // between reconnects.
    let camera = Arc::new(Mutex::new(camera));

    let audio_info = capture::audio::start_audio_capture(Arc::clone(&ring_buffer))?;
    let audio_sample_rate = audio_info.sample_rate;
    let audio_channels = audio_info.channels;
    let _audio_stream = audio_info.stream;

    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_handler = Arc::clone(&shutdown);

    ctrlc::set_handler(move || {
        log::info!("shutdown requested; finishing any in-progress recording before exit");
        shutdown_handler.store(true, Ordering::SeqCst);
    })
    .context("failed to register shutdown handler")?;

    let show_preview = config.preview;

    // The active recording, if any, is shared between the detection worker
    // (which decides when to start/stop it and records YOLO confirmations
    // into it) and a dedicated recording-writer thread (which only writes
    // frames/audio to it on a steady clock). Splitting these apart matters
    // because motion-gate + YOLO inference can take far longer than one
    // video frame interval; if the same loop iteration that runs inference
    // also writes the frame, frames land in the encoder's stdin at
    // irregular, stretched-out intervals while ffmpeg's fixed `-framerate`
    // assumes uniform spacing, which plays back as a stutter in the
    // recorded clip.
    let active_event: Arc<Mutex<ActiveEvent>> = Arc::new(Mutex::new(ActiveEvent::None));

    // Motion detection, YOLO confirmation, and the recording lifecycle run on
    // a dedicated worker thread at their own (much slower) pace, since YOLO
    // inference can take far longer than a single video frame interval. The
    // preview window stays on the main thread (highgui's GUI event loop isn't
    // safe to drive from a background thread) and runs its own fast display
    // loop pulling directly from the ring buffer, so a slow detection pass
    // never causes the visible feed to stutter.
    // Signaled by the writer thread after its post-shutdown last-chance
    // drain completes; the detection loop's shutdown path waits on this
    // before finishing the active event, so the writer's final drain is
    // guaranteed to land before the clip is finalized instead of racing it.
    let writer_drained = Arc::new(WriterDrained::default());

    let worker_shutdown = Arc::clone(&shutdown);
    let worker_ring_buffer = Arc::clone(&ring_buffer);
    let worker_active_event = Arc::clone(&active_event);
    let worker_writer_drained = Arc::clone(&writer_drained);
    let worker_camera = DetectionCamera {
        handle: Arc::clone(&camera),
        device: config.camera_device.clone(),
    };

    let worker_audio = AudioParams {
        sample_rate: audio_sample_rate,
        channels: audio_channels,
    };

    let worker_handle = thread::spawn(move || {
        if let Err(err) = run_detection_loop(
            config,
            worker_ring_buffer,
            worker_shutdown,
            worker_active_event,
            worker_writer_drained,
            worker_audio,
            worker_camera,
        ) {
            log::error!("detection worker exited with error: {err:?}");
        }
    });

    let writer_shutdown = Arc::clone(&shutdown);
    let writer_ring_buffer = Arc::clone(&ring_buffer);
    let writer_active_event = Arc::clone(&active_event);
    let writer_handle = thread::spawn(move || {
        run_recording_writer_loop(
            writer_ring_buffer,
            writer_active_event,
            writer_shutdown,
            writer_drained,
        );
    });

    log::info!("motioncap started; watching for motion");

    run_preview_loop(&ring_buffer, &shutdown, show_preview)?;

    worker_handle.join().expect("detection worker panicked");
    writer_handle.join().expect("recording writer panicked");

    Ok(())
}

/// Seeds a `Pending` event's pre-buffer (promoting it to `Active`) and drains
/// any newly-captured frames/audio into whichever event is active, if any.
/// Shared between the writer thread's normal poll cadence and its
/// shutdown path (see `run_recording_writer_loop`), so a shutdown that lands
/// mid-inference in the detection loop still gets one last drain instead of
/// silently dropping trailing footage the writer thread would otherwise
/// never pick up.
///
/// The lock is deliberately held across the seed/drain calls below, not just
/// the state-swap: `active_event` must reflect "a recording is in flight"
/// continuously for the detection loop's shutdown path (which calls `take()`
/// and finishes the event) and its `is_some()` check to observe consistent
/// state. Releasing it mid-drain would open a window where a concurrent
/// shutdown fails to finalize the in-flight clip.
#[allow(
    clippy::significant_drop_tightening,
    reason = "guard must stay held across seed/drain so shutdown's take() can't race an in-flight event"
)]
fn seed_and_drain_active_event(ring_buffer: &Mutex<RingBuffer>, active_event: &Mutex<ActiveEvent>) {
    let mut guard = active_event.lock().expect("active event lock poisoned");

    let taken = std::mem::replace(&mut *guard, ActiveEvent::None);

    match taken {
        ActiveEvent::Pending(mut pending) => {
            if let Err(err) = pending.event.seed(&pending.pre_frames, &pending.pre_audio) {
                log::error!("failed to seed pre-buffer into new recording: {err:?}");
            }
            *guard = ActiveEvent::Active(pending.event);
        }
        other => *guard = other,
    }

    let Some(event) = guard.as_recording_mut() else {
        return;
    };

    if let Err(err) = event.drain_frames(ring_buffer) {
        log::error!("failed to drain frames into active recording: {err:?}");
    }

    if let Err(err) = event.drain_audio(ring_buffer) {
        log::error!("failed to drain audio into active recording: {err:?}");
    }
}

/// Writes frames/audio into the active recording (if any) on a steady clock,
/// independent of how long motion-gate/YOLO evaluation takes in the
/// detection loop. This is what keeps recorded clips at a uniform frame
/// rate even while YOLO inference is running on the same tick elsewhere.
#[allow(
    clippy::needless_pass_by_value,
    reason = "Arc clones are moved into a spawned 'static thread closure, so they must be owned here"
)]
fn run_recording_writer_loop(
    ring_buffer: Arc<Mutex<RingBuffer>>,
    active_event: Arc<Mutex<ActiveEvent>>,
    shutdown: Arc<AtomicBool>,
    writer_drained: Arc<WriterDrained>,
) {
    loop {
        if shutdown.load(Ordering::SeqCst) {
            // The detection loop may still be blocked in YOLO inference when
            // shutdown fires and won't reach its own shutdown check (which
            // is what finishes the clip) until that inference pass
            // completes. Draining once more here before exiting means
            // whatever the camera/mic captured up to this instant still
            // makes it into the clip instead of being silently dropped from
            // the tail end of the recording. `writer_drained` is signaled
            // strictly after this drain completes, and `finish_event_on_shutdown`
            // waits on it before taking/finishing the event, so this drain is
            // guaranteed to land before the clip is finalized rather than
            // racing it.
            seed_and_drain_active_event(&ring_buffer, &active_event);
            writer_drained.signal();
            return;
        }

        thread::sleep(RECORDING_POLL_INTERVAL);

        seed_and_drain_active_event(&ring_buffer, &active_event);
    }
}

/// Displays the latest ring-buffer frame in a live preview window, if
/// `--preview` was passed. Runs on the main thread since `OpenCV`'s highgui
/// event loop isn't safe to drive from a background thread.
fn run_preview_loop(
    ring_buffer: &Arc<Mutex<RingBuffer>>,
    shutdown: &Arc<AtomicBool>,
    show_preview: bool,
) -> Result<()> {
    let mut preview = if show_preview {
        Some(PreviewWindow::open()?)
    } else {
        None
    };

    loop {
        if shutdown.load(Ordering::SeqCst) {
            return Ok(());
        }

        thread::sleep(PREVIEW_POLL_INTERVAL);

        let Some(preview) = preview.as_mut() else {
            continue;
        };

        let latest_frame = {
            let buf = ring_buffer.lock().expect("ring buffer lock poisoned");
            buf.latest_frame().map(|f| f.image.clone())
        };

        if let Some(frame) = latest_frame {
            preview.show(&frame)?;
        }
    }
}

/// Closes the active recording if either close condition is met: the camera
/// has stalled (see `RecordingEvent::camera_stalled`), which the motion gate
/// can never detect on its own since a stalled camera means no new frames
/// ever reach it to evaluate; or the post-buffer quiet window has elapsed
/// with no fresh trigger. No-ops if neither condition holds.
///
/// Takes the `MutexGuard` by value so it can be dropped before `finish`
/// (which waits on ffmpeg) runs -- the lock must not be held across that.
fn close_event_if_done(
    mut guard: MutexGuard<'_, ActiveEvent>,
    post_buffer: Duration,
) -> Result<()> {
    let Some(event) = guard.as_recording_mut() else {
        return Ok(());
    };

    let stalled = event.camera_stalled();
    let quiet_timed_out = event.quiet_for() >= post_buffer;

    if stalled || quiet_timed_out {
        let event = guard.take().expect("checked Some above");
        drop(guard);
        event.finish()?;

        if stalled {
            log::warn!("recording closed: camera stopped delivering frames");
        } else {
            log::info!("recording closed");
        }
    }

    Ok(())
}

/// Tracks the most recent frame timestamp `run_detection_loop` has evaluated,
/// and how long that timestamp has been unchanged, to detect a stalled camera
/// before any recording has started (see `frame_liveness_advanced`).
struct FrameLiveness {
    /// The last frame timestamp actually evaluated.
    timestamp: std::time::Instant,
    /// When `timestamp` was first observed to still be the latest frame.
    unchanged_since: std::time::Instant,
    /// Whether the stall warning has already been logged for this
    /// `timestamp`, so a still-stalled camera logs once per episode instead
    /// of once per poll.
    warned: bool,
}

/// Updates `last_seen` for a newly-polled `latest_frame` timestamp and
/// reports whether the loop should proceed with it. Returns `false` once the
/// same timestamp has recurred for `recorder::MAX_FRAME_STALL` -- i.e.
/// `latest_frame()` is returning a frame the camera delivered a while ago,
/// not a fresh one, which otherwise would get silently re-run through
/// motion/YOLO on every poll and cascade into duplicate recordings. A
/// same-timestamp recurrence shorter than that is ordinary jitter between
/// polls and camera delivery, not a stall, so it's allowed through unlogged.
/// The stall is logged once (not once per poll) when it's first detected.
fn frame_liveness_advanced(
    last_seen: &mut Option<FrameLiveness>,
    frame_timestamp: std::time::Instant,
) -> bool {
    let now = std::time::Instant::now();

    match last_seen {
        Some(seen) if seen.timestamp == frame_timestamp => {
            if now.duration_since(seen.unchanged_since) < recorder::MAX_FRAME_STALL {
                return true;
            }
            if !seen.warned {
                log::warn!(
                    "camera appears stalled: no new frame since {:?}; skipping detection ticks until it recovers",
                    seen.timestamp
                );
                seen.warned = true;
            }
            false
        }
        _ => {
            *last_seen = Some(FrameLiveness {
                timestamp: frame_timestamp,
                unchanged_since: now,
                warned: false,
            });
            true
        }
    }
}

/// If the current stall (tracked by `last_seen`) has persisted past
/// `CAMERA_RECONNECT_STALL`, tears down and rebuilds the capture stream (see
/// `capture::camera::reconnect_camera_stream`), respecting
/// `CAMERA_RECONNECT_COOLDOWN` between attempts so a camera that stays absent
/// doesn't get reopened on every single poll tick while it's gone.
/// `last_reconnect_attempt` is updated on every attempt (success or failure),
/// so the cooldown also applies after a success -- if the rebuilt stream
/// stalls again immediately, this still waits out the cooldown rather than
/// retrying in a tight loop.
///
/// Returns `true` on a successful rebuild, so the caller can reset
/// `last_seen`: the old stream's last timestamp is meaningless to compare
/// the freshly rebuilt stream's frames against.
fn maybe_reconnect_camera(
    last_seen: Option<&FrameLiveness>,
    last_reconnect_attempt: &mut Option<std::time::Instant>,
    camera: &Mutex<nokhwa::threaded::CallbackCamera>,
    camera_device: Option<&std::path::Path>,
    ring_buffer: &Arc<Mutex<RingBuffer>>,
) -> bool {
    let Some(seen) = last_seen else {
        return false;
    };

    let now = std::time::Instant::now();
    
    if now.duration_since(seen.unchanged_since) < CAMERA_RECONNECT_STALL {
        return false;
    }

    if let Some(attempted_at) = last_reconnect_attempt
        && now.duration_since(*attempted_at) < CAMERA_RECONNECT_COOLDOWN
    {
        return false;
    }

    *last_reconnect_attempt = Some(now);

    log::warn!(
        "camera has been stalled for over {CAMERA_RECONNECT_STALL:?}; attempting to rebuild the capture stream"
    );

    let rebuilt = capture::camera::reconnect_camera_stream(camera_device, Arc::clone(ring_buffer));

    match rebuilt {
        Ok(new_camera) => {
            *camera.lock().expect("camera lock poisoned") = new_camera;
            log::info!("camera stream rebuilt successfully");
            true
        }
        Err(err) => {
            log::error!("failed to rebuild camera stream: {err:?}");
            false
        }
    }
}

/// Runs the motion gate and (on trip) YOLO confirmation against `frame` for
/// an already-active recording, records the result into its sidecar, and
/// closes the event if either close condition in `close_event_if_done` is
/// met. Kept separate from `run_detection_loop`'s no-active-event path since
/// the two have no logic in common beyond polling the same frame.
#[allow(
    clippy::significant_drop_tightening,
    reason = "guard is moved into close_event_if_done, which drops it itself before the slow finish() call"
)]
fn evaluate_active_event(
    config: &Config,
    motion_gate: &mut MotionGate,
    detector: &mut Detector,
    active_event: &Arc<Mutex<ActiveEvent>>,
    frame: &image::RgbImage,
    frame_timestamp: std::time::Instant,
    post_buffer: Duration,
) -> Result<()> {
    let motion = motion_gate.evaluate(frame)?;

    // `detector.detect` runs without holding the event lock since YOLO
    // inference is the slow step; the recording writer thread must be free
    // to keep writing frames/audio at a steady pace while this runs, not
    // blocked waiting on this lock.
    let confirmed = if motion.tripped {
        let detections = detector.detect(frame, config.detection_confidence)?;
        triggers::evaluate(detections)
    } else {
        None
    };

    let mut guard = active_event.lock().expect("active event lock poisoned");
    let Some(event) = guard.as_recording_mut() else {
        // Recording was closed elsewhere (e.g. shutdown) while inference was
        // running above, or the writer thread hasn't finished seeding it yet.
        return Ok(());
    };

    if motion.tripped {
        event.record_motion(motion.changed_ratio, frame_timestamp);

        if let Some(confirmed) = &confirmed {
            for d in confirmed {
                event.record_detection(d.class_name, d.confidence, frame_timestamp);
            }
        } else {
            // Motion continues but wasn't re-confirmed by YOLO on this exact
            // frame; still reset the quiet-window so a subject that briefly
            // stops moving doesn't get cut off early.
            event.touch();
        }
    }

    close_event_if_done(guard, post_buffer)
}

/// The shared camera handle `run_detection_loop` polls liveness against and,
/// on a prolonged stall, rebuilds via `maybe_reconnect_camera`. Bundled
/// together since `camera_device` must be re-passed to
/// `capture::camera::reconnect_camera_stream` on every reconnect attempt.
struct DetectionCamera {
    /// The live capture stream; swapped out in place on reconnect.
    handle: Arc<Mutex<nokhwa::threaded::CallbackCamera>>,
    /// The originally configured device (or `None` for auto-detect), reused
    /// unchanged on every reconnect attempt.
    device: Option<std::path::PathBuf>,
}

/// Audio stream parameters the recording writer needs to configure ffmpeg's
/// input, captured once at startup and passed through unchanged.
struct AudioParams {
    /// Sample rate of the captured audio stream, in Hz.
    sample_rate: u32,
    /// Number of audio channels in the captured stream.
    channels: u16,
}

/// Polls the ring buffer at `DETECTION_FRAME_RATE`, runs the motion gate and
/// (on trip) YOLO confirmation, and owns the recording lifecycle: starting a
/// new `ActiveEvent::Pending` on a confirmed detection and closing it once
/// the post-buffer quiet window elapses.
#[allow(
    clippy::needless_pass_by_value,
    reason = "config and Arc clones are moved into a spawned 'static thread closure, so they must be owned here"
)]
#[allow(
    clippy::significant_drop_tightening,
    reason = "guard is moved into close_event_if_done, which drops it itself before the slow finish() call"
)]
fn run_detection_loop(
    config: Config,
    ring_buffer: Arc<Mutex<RingBuffer>>,
    shutdown: Arc<AtomicBool>,
    active_event: Arc<Mutex<ActiveEvent>>,
    writer_drained: Arc<WriterDrained>,
    audio: AudioParams,
    camera: DetectionCamera,
) -> Result<()> {
    let post_buffer = Duration::from_secs(u64::from(config.post_buffer_secs));

    let mut motion_gate = MotionGate::new(config.motion_threshold)?;
    let mut detector = Detector::load(config.model_path(), config.force_cpu)?;

    // Tracks the last frame timestamp this loop actually evaluated, plus when
    // that tracking last changed, so a stalled camera (`latest_frame()` keeps
    // returning the same frame forever, e.g. after a USB drop) is detected
    // rather than silently re-run through motion/YOLO on every poll -- which
    // would otherwise cascade into an unbounded stream of duplicate
    // recordings each time the post-buffer window elapses.
    let mut last_frame_seen: Option<FrameLiveness> = None;
    // When the last reconnect attempt was made, so a camera that stays
    // stalled doesn't get the stream rebuilt on every single poll tick (see
    // `maybe_reconnect_camera` / `CAMERA_RECONNECT_COOLDOWN`).
    let mut last_reconnect_attempt: Option<std::time::Instant> = None;

    loop {
        if shutdown.load(Ordering::SeqCst) {
            finish_event_on_shutdown(&active_event, &writer_drained)?;
            return Ok(());
        }

        thread::sleep(DETECTION_POLL_INTERVAL);

        let latest_frame = {
            let buf = ring_buffer.lock().expect("ring buffer lock poisoned");

            buf.latest_frame().map(|f| (f.image.clone(), f.timestamp))
        };

        let Some((frame, frame_timestamp)) = latest_frame else {
            continue;
        };

        let frame_is_live = frame_liveness_advanced(&mut last_frame_seen, frame_timestamp);

        // Even when the polled frame is stale (camera stalled), the recording
        // lifecycle must still be checked every tick -- a stalled camera is
        // exactly the condition `camera_stalled` (via `close_event_if_done`)
        // exists to close, and the post-buffer quiet window is wall-clock-based
        // and must keep expiring regardless of whether fresh frames are arriving.
        if !frame_is_live {
            let guard = active_event.lock().expect("active event lock poisoned");
            close_event_if_done(guard, post_buffer)?;

            if maybe_reconnect_camera(
                last_frame_seen.as_ref(),
                &mut last_reconnect_attempt,
                &camera.handle,
                camera.device.as_deref(),
                &ring_buffer,
            ) {
                last_frame_seen = None;
            }

            continue;
        }

        let has_active_event = active_event
            .lock()
            .expect("active event lock poisoned")
            .is_some();

        if has_active_event {
            evaluate_active_event(
                &config,
                &mut motion_gate,
                &mut detector,
                &active_event,
                &frame,
                frame_timestamp,
                post_buffer,
            )?;
            continue;
        }

        let motion = motion_gate.evaluate(&frame)?;

        log::trace!("frame received; motion_tripped={}", motion.tripped);

        if !motion.tripped {
            continue;
        }

        let detections = detector.detect(&frame, config.detection_confidence)?;

        log::trace!(
            "motion tripped; {} detections above threshold",
            detections.len()
        );

        let Some(confirmed) = triggers::evaluate(detections) else {
            continue;
        };

        let mut classes: Vec<&str> = confirmed.iter().map(|d| d.class_name).collect();

        classes.sort_unstable();
        classes.dedup();

        let (pre_frames, pre_audio) = {
            let buf = ring_buffer.lock().expect("ring buffer lock poisoned");
            buf.snapshot()
        };

        let Some(first_pre_frame) = pre_frames.first() else {
            // No buffered frames yet (e.g. trigger fired immediately at
            // startup, before the camera has produced anything); skip this
            // trigger rather than starting a recording with no video.
            continue;
        };

        let (width, height) = first_pre_frame.image.dimensions();
        let clip_timeline_start = first_pre_frame.timestamp;

        let started_at = chrono::Local::now();
        let path = clip_path(&config.output_dir, started_at, &classes)?;
        let mut event = RecordingEvent::start(RecordingEventParams {
            final_clip_path: path,
            output_dir: config.output_dir.clone(),
            started_at,
            width,
            height,
            frame_rate: RECORDING_FRAME_RATE,
            audio_sample_rate: audio.sample_rate,
            audio_channels: audio.channels,
            clip_timeline_start,
        })?;

        event.record_motion(motion.changed_ratio, frame_timestamp);

        for d in &confirmed {
            event.record_detection(d.class_name, d.confidence, frame_timestamp);
        }

        log::info!("recording started: {classes:?}");

        *active_event.lock().expect("active event lock poisoned") =
            ActiveEvent::Pending(PendingEvent {
                event,
                pre_frames,
                pre_audio,
            });
    }
}
