//! motioncap: webcam-based security motion capture. See the project guidance
//! file and `docs/adr/` for architecture and design-decision context.

/// Rolling pre-buffer of recent frames/audio (see `RingBuffer`).
mod buffer;
/// Camera and audio capture callbacks.
mod capture;
/// Pure timing/bookkeeping state for a single recorded clip.
mod clip_state;
/// CLI argument parsing.
mod config;
/// Repeat-sighting confirmation gate for YOLO detections.
mod confirmation;
/// YOLO object-detection inference.
mod detect;
/// Recording-event state machine and shutdown/close/seed-drain lifecycle.
mod event_lifecycle;
/// ffmpeg subprocess helpers (video encoder, audio mux, resampling).
mod ffmpeg;
/// Logging setup (`init_logging`, `TeeWriter`).
mod logging;
/// Background-subtraction motion gate.
mod motion;
/// Shared `OpenCV` conversion helpers.
mod opencv_utils;
/// Output file/folder naming.
mod paths;
/// Opt-in live preview window.
mod preview;
/// Camera liveness/stall detection and reconnect logic.
mod reconnect;
/// Recording lifecycle and ffmpeg-backed encoding.
mod recorder;
/// Clip `.json` sidecar output shapes (ADR 4).
mod sidecar;
/// Startup dependency checks.
mod startup;
/// Test fixtures shared across more than one module's test suite.
mod test_support;
/// Per-poll start/extend trigger decisions driving the recording lifecycle.
mod triggering;
/// YOLO-detection-to-trigger evaluation.
mod triggers;
/// Shutdown handshake primitive between the detection and writer threads.
mod writer_drained;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};

use buffer::RingBuffer;
use config::Config;
use confirmation::PendingConfirmation;
use detect::Detector;
use event_lifecycle::{
    ActiveEvent, close_event_if_done, finish_event_on_shutdown, seed_and_drain_active_event,
};
use logging::init_logging;
use motion::MotionGate;
use preview::PreviewWindow;
use reconnect::{
    DetectionCamera, FrameLiveness, frame_liveness_advanced, maybe_reconnect_camera,
    reset_liveness_after_reconnect,
};
use triggering::{
    AudioParams, RECORDING_POLL_INTERVAL, evaluate_active_event, try_start_recording,
};
use writer_drained::WriterDrained;

/// Motion-gate + YOLO evaluation cadence. Kept separate from the recording
/// frame rate below since inference cost doesn't scale down usefully at
/// higher polling rates. 15fps is plenty for deciding whether a subject is
/// still present.
const DETECTION_FRAME_RATE: u32 = 15;
/// Poll interval derived from `DETECTION_FRAME_RATE`.
pub(crate) const DETECTION_POLL_INTERVAL: Duration =
    Duration::from_millis(1000 / DETECTION_FRAME_RATE as u64);

/// Live preview window refresh rate (diagnostic only; see `preview.rs`).
const PREVIEW_FRAME_RATE: u32 = 30;
/// Poll interval derived from `PREVIEW_FRAME_RATE`.
const PREVIEW_POLL_INTERVAL: Duration = Duration::from_millis(1000 / PREVIEW_FRAME_RATE as u64);

/// Entry point. Opens the real camera/audio devices and spawns long-lived
/// worker threads, so it's not exercised by an automated test.
fn main() -> Result<()> {
    run()
}

/// Starts capture, the detection worker, the recording writer, and (if
/// `--preview` is set) the preview loop, then blocks on the preview loop
/// until shutdown. Called by `main`. Opens the real camera and audio devices
/// and spawns long-lived worker threads, none of which can run without
/// physical hardware, so it's not exercised by an automated test.
fn run() -> Result<()> {
    let config = config::parse_args();
    init_logging(&config.output_dir)?;

    startup::depcheck::check_dependencies(&config)?;

    let pre_buffer = Duration::from_secs(u64::from(config.pre_buffer_secs));
    let ring_buffer = Arc::new(Mutex::new(RingBuffer::new(pre_buffer)));

    let camera = capture::camera::start_camera_capture(
        config.camera_device.as_deref(),
        Arc::clone(&ring_buffer),
    )?;
    // Shared with the detection worker so it can rebuild the stream in place
    // when `CAMERA_RECONNECT_STALL` trips (see `FrameLiveness`); held here
    // too so the capture thread stays alive for the process lifetime even
    // between reconnects. Wrapped in `Option` so a reconnect can `take()` and
    // drop the old stream (releasing the device node) before opening the
    // replacement; see `maybe_reconnect_camera`.
    let camera = Arc::new(Mutex::new(Some(camera)));

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
    // video frame interval: if the same loop iteration that runs inference
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

/// Polls the ring buffer at `DETECTION_FRAME_RATE`, runs the motion gate and
/// (on trip) YOLO confirmation, and owns the recording lifecycle: starting a
/// new `ActiveEvent::Pending` on a confirmed detection and closing it once
/// the post-buffer quiet window elapses. Constructs a real `Detector::load`
/// (the YOLO model file + an ONNX Runtime session) unconditionally before
/// its loop body ever runs, so even exercising just its shutdown path would
/// require the same hardware/model dependency the `#[ignore]`d tests in
/// `detect.rs` already isolate as local-only; it's not exercised by an
/// automated test.
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
    // rather than silently re-run through motion/YOLO on every poll, which
    // would otherwise cascade into an unbounded stream of duplicate
    // recordings each time the post-buffer window elapses.
    let mut last_frame_seen: Option<FrameLiveness> = None;
    // When the last reconnect attempt was made, so a camera that stays
    // stalled doesn't get the stream rebuilt on every single poll tick (see
    // `maybe_reconnect_camera` / `CAMERA_RECONNECT_COOLDOWN`).
    let mut last_reconnect_attempt: Option<std::time::Instant> = None;
    // When the capture stream was last successfully rebuilt, so
    // `try_start_recording` can hold off starting a new recording until the
    // ring buffer has had a full `pre_buffer_secs` to refill from the
    // rebuilt stream (see `pre_buffer_ready`).
    let mut reconnected_at: Option<std::time::Instant> = None;
    // A first, unconfirmed living-thing sighting awaiting a second one to
    // start a recording (see `confirm_pending` / `PENDING_CONFIRMATION_WINDOW`).
    let mut pending_confirmation: Option<PendingConfirmation> = None;
    // The equivalent pending state for a recording already in progress (see
    // `evaluate_active_event`). Kept separate from `pending_confirmation`
    // rather than shared: they answer different questions (whether to start
    // a new recording vs. whether to trust a hit enough to extend one
    // already justified by an earlier confirmed detection), and sharing
    // state across that boundary would let a stale pre-recording sighting
    // spuriously confirm a hit against an event it had nothing to do with.
    let mut active_pending_confirmation: Option<PendingConfirmation> = None;

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
        // lifecycle must still be checked every tick: a stalled camera is
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
                reset_liveness_after_reconnect(&mut last_frame_seen);
                reconnected_at = Some(std::time::Instant::now());
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
                &mut active_pending_confirmation,
            )?;
            continue;
        }

        active_pending_confirmation = None;

        try_start_recording(
            &config,
            &mut motion_gate,
            &mut detector,
            &ring_buffer,
            &active_event,
            &audio,
            &frame,
            frame_timestamp,
            &mut pending_confirmation,
            reconnected_at,
        )?;
    }
}

/// Writes frames/audio into the active recording (if any) on a steady clock,
/// independent of how long motion-gate/YOLO evaluation takes in the
/// detection loop. This is what keeps recorded clips at a uniform frame
/// rate even while YOLO inference is running on the same tick elsewhere.
/// Testable up to its shutdown-check branch (a plain `AtomicBool`, no
/// hardware needed; see the tests below), but its steady-state loop body
/// waits on real wall-clock time (`thread::sleep`) and then drains real
/// ring-buffer/ffmpeg state, so that part is not exercised by an automated
/// test.
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
/// event loop isn't safe to drive from a background thread. Testable up to
/// its shutdown-check branch (see the tests below), but its steady-state
/// loop body waits on real wall-clock time and then drives `PreviewWindow`,
/// which needs a real `OpenCV` highgui display, so that part is not
/// exercised by an automated test.
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

#[cfg(test)]
mod tests {
    //! Unit tests for the loop functions that stay in this file. The
    //! recording-event state machine, confirmation gate, and camera
    //! reconnect logic have their own test modules in `event_lifecycle`,
    //! `confirmation`, and `reconnect` respectively.
    #![allow(
        clippy::unwrap_used,
        clippy::missing_panics_doc,
        reason = "test assertions favor unwrap for clarity; panics here fail the test, which is the intended behavior"
    )]

    use super::*;

    #[test]
    fn run_recording_writer_loop_drains_and_signals_on_immediate_shutdown() {
        let ring_buffer = Arc::new(Mutex::new(RingBuffer::new(Duration::from_secs(10))));
        let active_event = Arc::new(Mutex::new(ActiveEvent::None));
        let shutdown = Arc::new(AtomicBool::new(true));
        let writer_drained = Arc::new(WriterDrained::default());

        run_recording_writer_loop(
            ring_buffer,
            active_event,
            shutdown,
            Arc::clone(&writer_drained),
        );

        // Must not block: signal() was called before the function returned.
        writer_drained.wait();
    }

    #[test]
    fn run_preview_loop_returns_immediately_on_shutdown_without_preview() {
        let ring_buffer = Arc::new(Mutex::new(RingBuffer::new(Duration::from_secs(10))));
        let shutdown = Arc::new(AtomicBool::new(true));

        // show_preview=false skips PreviewWindow::open() entirely, so this
        // needs no display; only the shutdown-check branch is exercised.
        let result = run_preview_loop(&ring_buffer, &shutdown, false);

        assert!(result.is_ok());
    }
}
