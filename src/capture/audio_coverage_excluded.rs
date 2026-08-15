//! Holds `start_audio_capture`, the one function in `capture::audio` that
//! cannot be exercised by an automated test under any circumstances: every
//! line in it depends on a real default audio input device via `cpal`
//! (`default_host`/`default_input_device`/`default_input_config`/
//! `build_input_stream`/`stream.play`), which CI and many headless dev
//! sessions don't have. Split out of `audio.rs` (same convention as
//! `coverage_excluded.rs` at the crate root) so that file can reach genuine
//! 100% coverage instead of being held down to whatever fraction its one
//! untestable function happens to be, or excluded wholesale and losing the
//! ability to detect a regression in `samples_to_f32`/`sample_format_supported`,
//! which stay in `audio.rs` and are unit-tested there. See
//! `docs/adr/0006-coverage-exclusions.md`.

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use super::audio::{AudioStreamInfo, sample_format_supported, samples_to_f32};
use crate::buffer::RingBuffer;

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

    if !sample_format_supported(sample_format) {
        anyhow::bail!("unsupported audio sample format: {sample_format:?}");
    }

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
                    buf.push_audio(samples_to_f32(data));
                }
            },
            error_callback,
            None,
        ),
        SampleFormat::U16 => device.build_input_stream(
            stream_config,
            move |data: &[u16], _: &cpal::InputCallbackInfo| {
                if let Ok(mut buf) = buffer.lock() {
                    buf.push_audio(samples_to_f32(data));
                }
            },
            error_callback,
            None,
        ),
        // sample_format_supported already rejected anything other than
        // F32/I16/U16 above, so this arm is only reachable if that guard and
        // this match ever drift out of sync with each other.
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
