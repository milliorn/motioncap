//! Holds the two pieces of `capture::camera` that cannot be exercised by an
//! automated test under any circumstances: `auto_detect_camera_index`
//! (enumerates and probe-opens real cameras via
//! `nokhwa::query`/`nokhwa::Camera::new`) and all of `start_camera_capture`
//! (opens a real camera device via `CallbackCamera::new`/`open_stream`),
//! neither of which CI or many headless dev sessions have. Split out of
//! `camera.rs` (same convention as `audio_coverage_excluded.rs`) so that file
//! can reach genuine 100% coverage instead of being held down to whatever
//! fraction these two pieces happen to be. See
//! `docs/adr/0006-coverage-exclusions.md`.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::threaded::CallbackCamera;
use nokhwa::utils::{ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType};

use super::camera::resolve_pinned_camera_index;
use crate::buffer::RingBuffer;

/// Auto-selects a working camera when no device was pinned, by enumerating
/// candidates and actually trying to open each one, since a physical webcam
/// commonly exposes several `/dev/videoN` nodes that enumerate but fail to
/// open for capture (metadata/IR-sensor nodes).
pub(super) fn auto_detect_camera_index() -> Result<CameraIndex> {
    let cameras = nokhwa::query(ApiBackend::Auto).context("failed to enumerate cameras")?;

    for candidate in &cameras {
        let index = candidate.index().clone();
        let format =
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);

        match nokhwa::Camera::new(index.clone(), format) {
            Ok(_) => {
                log::info!("Auto-selected camera: {} ({index})", candidate.human_name());
                return Ok(index);
            }
            Err(err) => {
                log::debug!(
                    "skipping camera {} ({index}): {err}",
                    candidate.human_name()
                );
            }
        }
    }
    anyhow::bail!("no usable camera devices found")
}

/// Starts the webcam capture loop, decoding each incoming frame and pushing it
/// into the shared ring buffer. nokhwa manages its own capture thread
/// internally; the returned `CallbackCamera` must be kept alive for capture to
/// continue.
///
/// Also used to rebuild the stream from scratch after a stall that has
/// persisted well past ordinary jitter (see `main::CAMERA_RECONNECT_STALL`):
/// nokhwa's threaded v4l backend can enter a state where its internal capture
/// thread spins forever calling a failing `frame()` without ever surfacing an
/// error or exiting (the thread only checks a die flag between attempts, see
/// `nokhwa::threaded::camera_frame_thread_loop`). This happens with no
/// corresponding USB/kernel-level disconnect, so nokhwa itself never notices
/// and never recovers on its own. `CallbackCamera::stop_stream` cannot unstick
/// this: it only clears the inner `Camera`'s stream handle, not the spinning
/// background thread, so a subsequent `open_stream` on the *same* instance
/// just fails with "Stream Already Open"; the wedged thread outlives any
/// in-place restart attempt. The only way to actually replace it is to drop
/// the whole `CallbackCamera` (its `Drop` impl sets the thread's die flag)
/// and call this function again to construct a fresh one, exactly as at
/// startup.
pub fn start_camera_capture(
    device: Option<&Path>,
    buffer: Arc<Mutex<RingBuffer>>,
) -> Result<CallbackCamera> {
    let index = match resolve_pinned_camera_index(device)? {
        Some(index) => index,
        None => auto_detect_camera_index()?,
    };
    let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);

    let mut camera = CallbackCamera::new(index, format, move |raw_frame| {
        match raw_frame.decode_image::<RgbFormat>() {
            Ok(image) => {
                if let Ok(mut buf) = buffer.lock() {
                    buf.push_frame(image);
                }
            }
            Err(err) => log::warn!("failed to decode camera frame: {err}"),
        }
    })
    .context("failed to open camera")?;

    camera
        .open_stream()
        .context("failed to start camera stream")?;

    Ok(camera)
}
