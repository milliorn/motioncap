use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::threaded::CallbackCamera;
use nokhwa::utils::{ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType};

use crate::buffer::RingBuffer;

/// Resolves a configured camera device (e.g. `/dev/video0`) to a nokhwa
/// `CameraIndex`, or auto-selects a working camera if none was specified, so
/// the same binary works whether or not the user pinned a device.
///
/// On Linux, nokhwa's v4l backend expects `CameraIndex::Index(N)` (matching
/// `/dev/videoN`), not a path string. `CameraIndex::String` is reserved for
/// IP cameras. A physical webcam (e.g. the Logitech BRIO) commonly exposes
/// several `/dev/videoN` nodes, only one of which is the actual capture
/// device; the others are metadata/IR-sensor nodes that enumerate but fail to
/// open for capture. So auto-detection can't just take the first enumerated
/// camera; it must actually try opening each candidate and use the first
/// one that succeeds.
fn resolve_camera_index(device: Option<&Path>) -> Result<CameraIndex> {
    if let Some(path) = device {
        let name = path
            .to_str()
            .context("camera device path must be valid UTF-8")?;

        let index_str = name.trim_start_matches("/dev/video");

        let index: u32 = index_str
            .parse()
            .with_context(|| format!("expected a /dev/videoN path, got {name}"))?;

        return Ok(CameraIndex::Index(index));
    }

    // coverage: excluded from here to the end of the function, auto-detect
    // requires real camera hardware to enumerate and probe-open (nokhwa::query
    // + nokhwa::Camera::new), which CI and many headless dev sessions don't
    // have. See docs/adr/0006-coverage-exclusions.md.
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
// coverage: excluded, opens a real camera device (CallbackCamera::new +
// open_stream), which CI and many headless dev sessions don't have. See
// docs/adr/0006-coverage-exclusions.md.
pub fn start_camera_capture(
    device: Option<&Path>,
    buffer: Arc<Mutex<RingBuffer>>,
) -> Result<CallbackCamera> {
    let index = resolve_camera_index(device)?;
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

#[cfg(test)]
mod tests {
    //! Unit tests for `resolve_camera_index`'s pure `Some(path)` branch. The
    //! auto-detect branch (`nokhwa::query`/`nokhwa::Camera::new`) requires a
    //! real camera device and is left untested here (see
    //! `docs/adr/0006-coverage-exclusions.md`).
    #![allow(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        clippy::missing_panics_doc,
        reason = "test assertions favor unwrap/indexing for clarity; panics here fail the test, which is the intended behavior"
    )]

    use super::*;

    #[test]
    fn resolves_dev_video_path_to_matching_index() {
        let index = resolve_camera_index(Some(Path::new("/dev/video0"))).unwrap();
        assert_eq!(index, CameraIndex::Index(0));
    }

    #[test]
    fn resolves_dev_video_path_with_multi_digit_index() {
        let index = resolve_camera_index(Some(Path::new("/dev/video12"))).unwrap();
        assert_eq!(index, CameraIndex::Index(12));
    }

    #[test]
    fn rejects_non_numeric_suffix() {
        let result = resolve_camera_index(Some(Path::new("/dev/videoX")));
        assert!(result.is_err());
    }

    #[test]
    fn rejects_path_not_matching_dev_video_pattern() {
        // trim_start_matches leaves "/dev/webcam0" fully intact (no prefix
        // match), which then fails to parse as a u32.
        let result = resolve_camera_index(Some(Path::new("/dev/webcam0")));
        assert!(result.is_err());
    }
}
