use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::threaded::CallbackCamera;
use nokhwa::utils::{ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType};

use crate::buffer::RingBuffer;

/// Resolves a configured camera device (e.g. `/dev/video0`) to a nokhwa
/// `CameraIndex`, or falls back to the first available camera if none was
/// specified, so the same binary works whether or not the user pinned a device.
fn resolve_camera_index(device: Option<&Path>) -> Result<CameraIndex> {
    if let Some(path) = device {
        let name = path
            .to_str()
            .context("camera device path must be valid UTF-8")?;
        return Ok(CameraIndex::String(name.to_string()));
    }

    let cameras = nokhwa::query(ApiBackend::Auto).context("failed to enumerate cameras")?;
    let first = cameras
        .into_iter()
        .next()
        .context("no camera devices found")?;
    log::info!("Auto-selected camera: {}", first.human_name());
    Ok(first.index().clone())
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
