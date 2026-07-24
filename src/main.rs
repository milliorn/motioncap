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
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};

use buffer::{RingBuffer, TimestampedAudio, TimestampedFrame};
use config::Config;
use detect::Detector;
use motion::MotionGate;
use paths::clip_path;
use preview::PreviewWindow;
use recorder::RecordingEvent;

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

/// Starts capture, the detection worker, the recording writer, and (if
/// `--preview` is set) the preview loop, then blocks on the preview loop
/// until shutdown.
fn main() -> Result<()> {
    env_logger::init();
    let config = Config::parse_args();

    startup::check_dependencies(&config)?;

    let pre_buffer = Duration::from_secs(u64::from(config.pre_buffer_secs));
    let ring_buffer = Arc::new(Mutex::new(RingBuffer::new(pre_buffer)));

    let _camera = capture::camera::start_camera_capture(
        config.camera_device.as_deref(),
        Arc::clone(&ring_buffer),
    )?;

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
    let worker_shutdown = Arc::clone(&shutdown);
    let worker_ring_buffer = Arc::clone(&ring_buffer);
    let worker_active_event = Arc::clone(&active_event);
    let worker_handle = thread::spawn(move || {
        if let Err(err) = run_detection_loop(
            config,
            worker_ring_buffer,
            worker_shutdown,
            worker_active_event,
            audio_sample_rate,
            audio_channels,
        ) {
            log::error!("detection worker exited with error: {err:?}");
        }
    });

    let writer_shutdown = Arc::clone(&shutdown);
    let writer_ring_buffer = Arc::clone(&ring_buffer);
    let writer_active_event = Arc::clone(&active_event);
    let writer_handle = thread::spawn(move || {
        run_recording_writer_loop(writer_ring_buffer, writer_active_event, writer_shutdown);
    });

    log::info!("motioncap started; watching for motion");

    run_preview_loop(&ring_buffer, &shutdown, show_preview)?;

    worker_handle.join().expect("detection worker panicked");
    writer_handle.join().expect("recording writer panicked");

    Ok(())
}

/// Writes frames/audio into the active recording (if any) on a steady clock,
/// independent of how long motion-gate/YOLO evaluation takes in the
/// detection loop. This is what keeps recorded clips at a uniform frame
/// rate even while YOLO inference is running on the same tick elsewhere.
#[allow(
    clippy::needless_pass_by_value,
    reason = "Arc clones are moved into a spawned 'static thread closure, so they must be owned here"
)]
#[allow(
    clippy::significant_drop_tightening,
    reason = "guard must stay held across seed/drain so shutdown's take() can't race an in-flight event"
)]
fn run_recording_writer_loop(
    ring_buffer: Arc<Mutex<RingBuffer>>,
    active_event: Arc<Mutex<ActiveEvent>>,
    shutdown: Arc<AtomicBool>,
) {
    loop {
        if shutdown.load(Ordering::SeqCst) {
            return;
        }

        thread::sleep(RECORDING_POLL_INTERVAL);

        // The lock is deliberately held across the seed/drain calls below,
        // not just the state-swap: `active_event` must reflect "a recording
        // is in flight" continuously for the detection loop's shutdown path
        // (which calls `take()` and finishes the event) and its `is_some()`
        // check to observe consistent state. Releasing it mid-drain would
        // open a window where a concurrent shutdown fails to finalize the
        // in-flight clip.
        let mut guard = active_event.lock().expect("active event lock poisoned");

        let taken = std::mem::replace(&mut *guard, ActiveEvent::None);

        match taken {
            ActiveEvent::Pending(mut pending) => {
                if let Err(err) = pending.event.seed(
                    &pending.pre_frames,
                    &pending.pre_audio,
                    RECORDING_FRAME_RATE,
                ) {
                    log::error!("failed to seed pre-buffer into new recording: {err:?}");
                }
                *guard = ActiveEvent::Active(pending.event);
            }
            other => *guard = other,
        }

        let Some(event) = guard.as_recording_mut() else {
            continue;
        };

        if let Err(err) = event.drain_frames(&ring_buffer) {
            log::error!("failed to drain frames into active recording: {err:?}");
        }
        if let Err(err) = event.drain_audio(&ring_buffer) {
            log::error!("failed to drain audio into active recording: {err:?}");
        }
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

/// Polls the ring buffer at `DETECTION_FRAME_RATE`, runs the motion gate and
/// (on trip) YOLO confirmation, and owns the recording lifecycle: starting a
/// new `ActiveEvent::Pending` on a confirmed detection and closing it once
/// the post-buffer quiet window elapses.
#[allow(
    clippy::needless_pass_by_value,
    reason = "config and Arc clones are moved into a spawned 'static thread closure, so they must be owned here"
)]
fn run_detection_loop(
    config: Config,
    ring_buffer: Arc<Mutex<RingBuffer>>,
    shutdown: Arc<AtomicBool>,
    active_event: Arc<Mutex<ActiveEvent>>,
    audio_sample_rate: u32,
    audio_channels: u16,
) -> Result<()> {
    let post_buffer = Duration::from_secs(u64::from(config.post_buffer_secs));

    let mut motion_gate = MotionGate::new(config.motion_threshold)?;
    let mut detector = Detector::load(config.model_path(), config.force_cpu)?;

    loop {
        if shutdown.load(Ordering::SeqCst) {
            let event = active_event
                .lock()
                .expect("active event lock poisoned")
                .take();

            if let Some(event) = event {
                event.finish()?;
                log::info!("recording closed on shutdown");
            }

            return Ok(());
        }

        thread::sleep(DETECTION_POLL_INTERVAL);

        let latest_frame = {
            let buf = ring_buffer.lock().expect("ring buffer lock poisoned");
            buf.latest_frame().map(|f| f.image.clone())
        };

        let Some(frame) = latest_frame else {
            continue;
        };

        let has_active_event = active_event
            .lock()
            .expect("active event lock poisoned")
            .is_some();

        if has_active_event {
            let motion_tripped = motion_gate.evaluate(&frame)?;

            // `detector.detect` runs without holding the event lock since
            // YOLO inference is the slow step; the recording writer thread
            // must be free to keep writing frames/audio at a steady pace
            // while this runs, not blocked waiting on this lock.
            let confirmed = if motion_tripped {
                let detections = detector.detect(&frame, config.detection_confidence)?;
                triggers::evaluate(detections)
            } else {
                None
            };

            let mut guard = active_event.lock().expect("active event lock poisoned");
            let Some(event) = guard.as_recording_mut() else {
                // Recording was closed elsewhere (e.g. shutdown) while
                // inference was running above, or the writer thread hasn't
                // finished seeding it yet.
                continue;
            };

            if motion_tripped {
                if let Some(confirmed) = &confirmed {
                    for d in confirmed {
                        event.record_detection(d.class_name, d.confidence);
                    }
                } else {
                    // Motion continues but wasn't re-confirmed by YOLO on this
                    // exact frame; still reset the quiet-window so a subject
                    // that briefly stops moving doesn't get cut off early.
                    event.touch();
                }
            }

            if event.quiet_for() >= post_buffer {
                let event = guard.take().expect("checked Some above");
                drop(guard);
                event.finish()?;
                log::info!("recording closed");
            }
            continue;
        }

        let motion_tripped = motion_gate.evaluate(&frame)?;

        log::trace!("frame received; motion_tripped={motion_tripped}");

        if !motion_tripped {
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

        let Some((width, height)) = pre_frames.first().map(|f| f.image.dimensions()) else {
            // No buffered frames yet (e.g. trigger fired immediately at
            // startup, before the camera has produced anything); skip this
            // trigger rather than starting a recording with no video.
            continue;
        };

        let path = clip_path(&config.output_dir, chrono::Local::now(), &classes)?;
        let mut event = RecordingEvent::start(
            path,
            width,
            height,
            RECORDING_FRAME_RATE,
            audio_sample_rate,
            audio_channels,
        )?;

        for d in &confirmed {
            event.record_detection(d.class_name, d.confidence);
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
