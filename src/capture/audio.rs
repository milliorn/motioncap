use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, Stream};

use crate::buffer::RingBuffer;

/// The audio format the rest of the pipeline (ring buffer, recorder/muxer)
/// operates on: samples are always converted to interleaved f32 at capture
/// time, but the actual sample rate/channel count depend on the device's
/// default input config, so callers need those to configure ffmpeg's mux step
/// correctly instead of assuming a fixed rate.
pub struct AudioStreamInfo {
    pub stream: Stream,
    pub sample_rate: u32,
    pub channels: u16,
}

/// Opens the default audio input device and starts streaming samples into the
/// shared ring buffer. The returned `Stream` must be kept alive for capture to
/// continue (dropping it stops the stream), so ownership is handed to the caller.
///
/// The device's default input config can report any sample format (f32, i16,
/// or u16), not just f32, so the callback is chosen based on the actual
/// reported format and always converts samples to f32 before buffering, since
/// that's the format the ring buffer and recorder expect.
pub fn start_audio_capture(buffer: Arc<Mutex<RingBuffer>>) -> Result<AudioStreamInfo> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("no default audio input device found")?;
    let config = device
        .default_input_config()
        .context("failed to get default audio input config")?;

    let sample_rate = config.sample_rate();
    let channels = config.channels();
    let sample_format = config.sample_format();
    let stream_config = config.into();

    let error_callback = |err: cpal::Error| log::warn!("audio stream error: {err}");

    let stream = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if let Ok(mut buf) = buffer.lock() {
                    buf.push_audio(data.to_vec());
                }
            },
            error_callback,
            None,
        ),
        SampleFormat::I16 => device.build_input_stream(
            stream_config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                if let Ok(mut buf) = buffer.lock() {
                    buf.push_audio(data.iter().map(|&s| f32::from_sample(s)).collect());
                }
            },
            error_callback,
            None,
        ),
        SampleFormat::U16 => device.build_input_stream(
            stream_config,
            move |data: &[u16], _: &cpal::InputCallbackInfo| {
                if let Ok(mut buf) = buffer.lock() {
                    buf.push_audio(data.iter().map(|&s| f32::from_sample(s)).collect());
                }
            },
            error_callback,
            None,
        ),
        other => anyhow::bail!("unsupported audio sample format: {other:?}"),
    }
    .context("failed to build audio input stream")?;

    stream.play().context("failed to start audio stream")?;
    Ok(AudioStreamInfo {
        stream,
        sample_rate,
        channels,
    })
}
