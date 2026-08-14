//! Holds the three functions in this crate that cannot be exercised by an
//! automated test under any circumstances, split out from `main.rs` so that
//! file remains fully unit-testable and `cargo-llvm-cov`'s coverage gate can
//! stay at (true) 100% there. `cargo-llvm-cov` on stable Rust only supports
//! whole-*file* exclusion (`--ignore-filename-regex`); the nightly-only
//! `#[coverage(off)]` attribute, which would allow excluding individual
//! functions in place, was tried and confirmed unavailable on this crate's
//! stable toolchain (see `docs/adr/0006-coverage-exclusions.md`). Mixing a
//! handful of untestable functions into `main.rs` alongside dozens of
//! thoroughly-tested ones meant the only available exclusion mechanism was
//! all-or-nothing: either accept `main.rs`'s reported coverage sitting in the
//! 50s (correct but discouraging, and looks identical to "nobody tried") --
//! or exclude the whole file and lose the ability to detect a regression in
//! its real, tested logic. Isolating the untestable functions here instead keeps
//! `main.rs` at genuine 100% and makes exactly what's excluded, and why,
//! visible in one place.
//!
//! Each function below is untestable for a different concrete reason (not
//! "this seemed hard"):
//! - [`run`] performs top-level wiring: it opens the real camera and audio
//!   devices and spawns long-lived worker threads, none of which can run
//!   without physical hardware.
//! - [`init_logging`] calls `env_logger::Builder::init()`, which sets the
//!   process-global logger. That call panics if made more than once in a
//!   process, and `cargo test` runs the entire suite in one process, so it
//!   can only be invoked here, in the real `main` path, without an
//!   order-dependent risk of poisoning every other test's use of `log::`.
//! - [`run_detection_loop`] constructs a real `Detector::load` (the YOLO
//!   model file + an ONNX Runtime session) unconditionally before its loop
//!   body ever runs, so even exercising just its shutdown path would
//!   require the same hardware/model dependency the `#[ignore]`d tests in
//!   `detect.rs` already isolate as local-only.
//!
//! None of these functions contain meaningful decision logic of their own --
//! every branch `run_detection_loop`'s loop takes is a call into a function
//! defined in `main.rs` and unit-tested there (`try_start_recording`,
//! `evaluate_active_event`, `frame_liveness_advanced`, `maybe_reconnect_camera`,
//! `finish_event_on_shutdown`, ...). This module is deliberately kept to pure
//! sequencing/wiring for exactly that reason; if a function here starts
//! accumulating real logic, that logic belongs in `main.rs` where it can be
//! tested, not here.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};

use crate::buffer::RingBuffer;
use crate::config::Config;
use crate::detect::Detector;
use crate::motion::MotionGate;
use crate::{
    ActiveEvent, AudioParams, DETECTION_POLL_INTERVAL, DetectionCamera, FrameLiveness,
    LOG_FILE_NAME, PendingConfirmation, TeeWriter, WriterDrained, capture, close_event_if_done,
    evaluate_active_event, finish_event_on_shutdown, frame_liveness_advanced,
    maybe_reconnect_camera, reset_liveness_after_reconnect, run_preview_loop,
    run_recording_writer_loop, startup, try_start_recording,
};

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
        .target(env_logger::Target::Pipe(Box::new(TeeWriter::new(file))))
        .init();

    Ok(())
}

/// Starts capture, the detection worker, the recording writer, and (if
/// `--preview` is set) the preview loop, then blocks on the preview loop
/// until shutdown. Called by `main`.
pub fn run() -> Result<()> {
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
    // rather than silently re-run through motion/YOLO on every poll, which
    // would otherwise cascade into an unbounded stream of duplicate
    // recordings each time the post-buffer window elapses.
    let mut last_frame_seen: Option<FrameLiveness> = None;
    // When the last reconnect attempt was made, so a camera that stays
    // stalled doesn't get the stream rebuilt on every single poll tick (see
    // `maybe_reconnect_camera` / `CAMERA_RECONNECT_COOLDOWN`).
    let mut last_reconnect_attempt: Option<std::time::Instant> = None;
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
        )?;
    }
}
