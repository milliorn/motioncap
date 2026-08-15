use cpal::{Sample, SampleFormat, Stream};

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
/// `build_input_stream` callbacks (see `audio_coverage_excluded.rs`); kept
/// here, separate from that hardware-bound function, so this conversion is
/// unit-testable on plain sample data without needing a live cpal callback to
/// invoke it.
pub(super) fn samples_to_f32<S: Sample>(data: &[S]) -> Vec<f32>
where
    f32: cpal::FromSample<S>,
{
    data.iter().map(|&s| f32::from_sample(s)).collect()
}

/// Whether `format` is one this crate knows how to convert to `f32` (see
/// `samples_to_f32`/the `SampleFormat::F32` passthrough). Checked by
/// `start_audio_capture` before it commits to building a stream for it, so an
/// unsupported format fails with a clear error instead of a `match` that
/// silently can't be reached. Kept here, not in `audio_coverage_excluded.rs`,
/// since it's a pure decision over a `SampleFormat` value, not itself a
/// hardware call.
pub(super) const fn sample_format_supported(format: SampleFormat) -> bool {
    matches!(
        format,
        SampleFormat::F32 | SampleFormat::I16 | SampleFormat::U16
    )
}

#[cfg(test)]
mod tests {
    //! Unit tests for the pure sample-conversion/format-support logic that
    //! `start_audio_capture` relies on. `start_audio_capture` itself lives in
    //! `audio_coverage_excluded.rs`, not this file, since every line of it
    //! requires a real audio input device (see that module's doc comment and
    //! `docs/adr/0006-coverage-exclusions.md`).
    #![allow(
        clippy::indexing_slicing,
        reason = "test assertions favor indexing for clarity; panics here fail the test, which is the intended behavior"
    )]

    use super::*;

    #[test]
    fn samples_to_f32_converts_i16_to_normalized_f32() {
        let converted = samples_to_f32::<i16>(&[i16::MIN, 0, i16::MAX]);

        assert_eq!(converted.len(), 3);
        assert!((converted[0] - -1.0).abs() < f32::EPSILON);
        assert!((converted[1] - 0.0).abs() < f32::EPSILON);
        assert!((converted[2] - 1.0).abs() < 0.001);
    }

    #[test]
    fn samples_to_f32_converts_u16_to_normalized_f32() {
        let converted = samples_to_f32::<u16>(&[u16::MIN, u16::MAX / 2, u16::MAX]);

        assert_eq!(converted.len(), 3);
        assert!((converted[0] - -1.0).abs() < 0.001);
        assert!((converted[2] - 1.0).abs() < 0.001);
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
