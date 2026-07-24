use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::Stream;

use crate::buffer::RingBuffer;

/// Opens the default audio input device and starts streaming samples into the
/// shared ring buffer. The returned `Stream` must be kept alive for capture to
/// continue (dropping it stops the stream), so ownership is handed to the caller.
pub fn start_audio_capture(buffer: Arc<Mutex<RingBuffer>>) -> Result<Stream> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("no default audio input device found")?;
    let config = device
        .default_input_config()
        .context("failed to get default audio input config")?;

    let stream = device
        .build_input_stream(
            config.into(),
            move |data: &[f32], _info: &cpal::InputCallbackInfo| {
                if let Ok(mut buf) = buffer.lock() {
                    buf.push_audio(data.to_vec());
                }
            },
            |err| log::warn!("audio stream error: {err}"),
            None,
        )
        .context("failed to build audio input stream")?;

    stream.play().context("failed to start audio stream")?;
    Ok(stream)
}
