mod buffer;
mod capture;
mod config;
mod detect;
mod motion;
mod paths;
mod preview;
mod recorder;
mod startup;
mod triggers;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};

use buffer::RingBuffer;
use config::Config;
use detect::Detector;
use motion::MotionGate;
use paths::clip_path;
use preview::PreviewWindow;
use recorder::RecordingEvent;

const DETECTION_FRAME_RATE: u32 = 15;
const DETECTION_POLL_INTERVAL: Duration = Duration::from_millis(1000 / DETECTION_FRAME_RATE as u64);
const PREVIEW_FRAME_RATE: u32 = 30;
const PREVIEW_POLL_INTERVAL: Duration = Duration::from_millis(1000 / PREVIEW_FRAME_RATE as u64);

fn main() -> Result<()> {
    env_logger::init();
    let config = Config::parse_args();

    startup::check_dependencies(&config)?;

    let pre_buffer = Duration::from_secs(config.pre_buffer_secs as u64);
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

    // Motion detection, YOLO confirmation, and the recording lifecycle run on
    // a dedicated worker thread at their own (much slower) pace, since YOLO
    // inference can take far longer than a single video frame interval. The
    // preview window stays on the main thread (highgui's GUI event loop isn't
    // safe to drive from a background thread) and runs its own fast display
    // loop pulling directly from the ring buffer, so a slow detection pass
    // never causes the visible feed to stutter.
    let worker_shutdown = Arc::clone(&shutdown);
    let worker_ring_buffer = Arc::clone(&ring_buffer);
    let worker_handle = thread::spawn(move || {
        if let Err(err) = run_detection_loop(
            config,
            worker_ring_buffer,
            worker_shutdown,
            audio_sample_rate,
            audio_channels,
        ) {
            log::error!("detection worker exited with error: {err:?}");
        }
    });

    log::info!("motioncap started; watching for motion");
    
    run_preview_loop(&ring_buffer, &shutdown, show_preview)?;

    worker_handle.join().expect("detection worker panicked");
    Ok(())
}

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

fn run_detection_loop(
    config: Config,
    ring_buffer: Arc<Mutex<RingBuffer>>,
    shutdown: Arc<AtomicBool>,
    audio_sample_rate: u32,
    audio_channels: u16,
) -> Result<()> {
    let post_buffer = Duration::from_secs(config.post_buffer_secs as u64);

    let mut motion_gate = MotionGate::new(config.motion_threshold)?;
    let mut detector = Detector::load(config.model_path(), config.force_cpu)?;
    let mut active_event: Option<RecordingEvent> = None;

    loop {
        if shutdown.load(Ordering::SeqCst) {
            if let Some(event) = active_event.take() {
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

        if let Some(event) = active_event.as_mut() {
            event.write_frame(&frame)?;
            event.drain_audio(&ring_buffer)?;

            let motion_tripped = motion_gate.evaluate(&frame)?;

            if motion_tripped {
                let detections = detector.detect(&frame, config.detection_confidence)?;
                if let Some(confirmed) = triggers::evaluate(detections) {
                    for d in &confirmed {
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
                let event = active_event.take().expect("checked Some above");
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

        log::trace!("motion tripped; {} detections above threshold", detections.len());

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

        let path = clip_path(&config.output_dir, chrono::Local::now(), &classes)?;
        let mut event = RecordingEvent::start(
            path,
            pre_frames,
            pre_audio,
            DETECTION_FRAME_RATE,
            audio_sample_rate,
            audio_channels,
        )?;

        for d in &confirmed {
            event.record_detection(d.class_name, d.confidence);
        }

        log::info!("recording started: {:?}", classes);
        
        active_event = Some(event);
    }
}
