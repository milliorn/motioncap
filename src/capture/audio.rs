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
    /// The live cpal stream; must be kept alive for capture to continue.
    pub stream: Stream,
    /// The input device's actual sample rate, needed to configure ffmpeg's mux step.
    pub sample_rate: u32,
    /// The input device's actual channel count, needed to configure ffmpeg's mux step.
    pub channels: u16,
}

/// Converts a buffer of `i16`/`u16` samples to interleaved `f32`, the format
/// the ring buffer and recorder expect. Used by `start_audio_capture`'s
/// `build_input_stream` callbacks; kept as its own function so this
/// conversion is unit-testable on plain sample data without needing a live
/// cpal callback to invoke it.
fn samples_to_f32<S: Sample>(data: &[S]) -> Vec<f32>
where
    f32: cpal::FromSample<S>,
{
    data.iter().map(|&s| f32::from_sample(s)).collect()
}

/// Whether `format` is one this crate knows how to convert to `f32` (see
/// `samples_to_f32`/the `SampleFormat::F32` passthrough). Checked by
/// `start_audio_capture` before it commits to building a stream for it, so an
/// unsupported format fails with a clear error instead of a `match` that
/// silently can't be reached.
const fn sample_format_supported(format: SampleFormat) -> bool {
    matches!(
        format,
        SampleFormat::F32 | SampleFormat::I16 | SampleFormat::U16
    )
}

/// Opens the default audio input device and starts streaming samples into the
/// shared ring buffer. The returned `Stream` must be kept alive for capture to
/// continue (dropping it stops the stream), so ownership is handed to the caller.
///
/// The device's default input config can report any sample format (f32, i16,
/// or u16), not just f32, so the callback is chosen based on the actual
/// reported format and always converts samples to f32 before buffering, since
/// that's the format the ring buffer and recorder expect. Every line depends
/// on a real default audio input device via cpal
/// (`default_host`/`default_input_device`/`default_input_config`/
/// `build_input_stream`/`stream.play`), which CI and many headless dev
/// sessions don't have, so this function is not covered by an automated test.
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

#[cfg(test)]
mod tests {
    //! Unit tests for the pure sample-conversion/format-support logic that
    //! `start_audio_capture` relies on. `start_audio_capture` itself requires
    //! a real audio input device and is left untested here.
    #![allow(
        clippy::indexing_slicing,
        reason = "test assertions favor indexing for clarity; panics here fail the test, which is the intended behavior"
    )]

    use super::*;

    /// Tolerance for comparing a converted sample against its expected
    /// normalized value; wider than `f32::EPSILON` to absorb the rounding
    /// `i16`/`u16` -> `f32` conversion introduces, without being loose enough
    /// to mask a real conversion bug.
    const NORMALIZED_SAMPLE_TOLERANCE: f32 = 0.001;

    /// Divisor used to derive `u16`'s exact midpoint sample value from its
    /// `MAX`, for a normalized-to-zero conversion test case.
    const U16_MIDPOINT_DIVISOR: u16 = 2;

    #[test]
    fn samples_to_f32_converts_i16_to_normalized_f32() {
        let converted = samples_to_f32::<i16>(&[i16::MIN, 0, i16::MAX]);

        assert_eq!(converted.len(), 3);
        assert!((converted[0] - -1.0).abs() < f32::EPSILON);
        assert!((converted[1] - 0.0).abs() < f32::EPSILON);
        assert!((converted[2] - 1.0).abs() < NORMALIZED_SAMPLE_TOLERANCE);
    }

    #[test]
    fn samples_to_f32_converts_u16_to_normalized_f32() {
        let converted =
            samples_to_f32::<u16>(&[u16::MIN, u16::MAX / U16_MIDPOINT_DIVISOR, u16::MAX]);

        assert_eq!(converted.len(), 3);
        assert!((converted[0] - -1.0).abs() < NORMALIZED_SAMPLE_TOLERANCE);
        assert!((converted[2] - 1.0).abs() < NORMALIZED_SAMPLE_TOLERANCE);
    }

    #[test]
    fn samples_to_f32_empty_input_yields_empty_output() {
        let converted = samples_to_f32::<i16>(&[]);
        assert!(converted.is_empty());
    }

    #[test]
    fn sample_format_supported_true_for_f32_i16_u16() {
        assert!(sample_format_supported(SampleFormat::F32));
        assert!(sample_format_supported(SampleFormat::I16));
        assert!(sample_format_supported(SampleFormat::U16));
    }

    #[test]
    fn sample_format_supported_false_for_other_formats() {
        assert!(!sample_format_supported(SampleFormat::I8));
        assert!(!sample_format_supported(SampleFormat::U8));
    }
}
