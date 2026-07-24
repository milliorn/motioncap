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
/// `/dev/videoN`), not a path string -- `CameraIndex::String` is reserved for
/// IP cameras. A physical webcam (e.g. the Logitech BRIO) commonly exposes
/// several `/dev/videoN` nodes, only one of which is the actual capture
/// device; the others are metadata/IR-sensor nodes that enumerate but fail to
/// open for capture. So auto-detection can't just take the first enumerated
/// camera -- it must actually try opening each candidate and use the first
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

    let cameras = nokhwa::query(ApiBackend::Auto).context("failed to enumerate cameras")?;

    for candidate in &cameras {
        let index = candidate.index().clone();
        let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
        
        match nokhwa::Camera::new(index.clone(), format) {
            Ok(_) => {
                log::info!("Auto-selected camera: {} ({index})", candidate.human_name());
                return Ok(index);
            }
            Err(err) => {
                log::debug!("skipping camera {} ({index}): {err}", candidate.human_name());
            }
        }
    }
    anyhow::bail!("no usable camera devices found")
}

/// Starts the webcam capture loop, decoding each incoming frame and pushing it
/// into the shared ring buffer. nokhwa manages its own capture thread
/// internally; the returned `CallbackCamera` must be kept alive for capture to
/// continue.
pub fn start_camera_capture(
    device: Option<&Path>,
    buffer: Arc<Mutex<RingBuffer>>,
) -> Result<CallbackCamera> {
    let index = resolve_camera_index(device)?;
    let format =
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);

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
