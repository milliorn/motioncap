use anyhow::{Context, Result};
use image::RgbImage;
// `Mat` ("matrix") is OpenCV's own core image/array type, not a Rust or std
// convention: https://docs.opencv.org/4.x/d3/d63/classcv_1_1Mat.html
use opencv::core::{Mat, MatTraitConst, ToInputArray};
use opencv::video::{BackgroundSubtractorTrait, create_background_subtractor_mog2};

use crate::opencv_utils::rgb_image_to_bgr_mat;

/// Background-subtraction motion gate (ADR 2). Its only job is deciding
/// whether whole-frame motion exceeded the configured threshold; recording
/// itself is only ever triggered by a subsequent confirmed YOLO
/// classification, never by this gate alone.
pub struct MotionGate {
    /// The `OpenCV` MOG2 background-subtraction model.
    subtractor: opencv::core::Ptr<opencv::video::BackgroundSubtractorMOG2>,
    /// Minimum changed-pixel ratio (0.0-1.0) for `evaluate` to report motion.
    threshold: f32,
}

/// The result of one `MotionGate::evaluate` call.
pub struct MotionReading {
    /// Fraction of pixels (0.0-1.0) the background model marked as changed.
    pub changed_ratio: f32,
    /// Whether `changed_ratio` exceeded the gate's configured threshold.
    pub tripped: bool,
}

impl MotionGate {
    /// Creates a motion gate with a fresh MOG2 background model.
    pub fn new(threshold: f32) -> Result<Self> {
        let subtractor = create_background_subtractor_mog2(500, 16.0, true)
            .context("failed to create MOG2 background subtractor")?;
        Ok(Self {
            subtractor,
            threshold,
        })
    }

    /// Feeds one frame through the background model and reports the
    /// changed-pixel ratio alongside whether it exceeded the configured
    /// threshold. Callers that need to log/record motion activity (not
    /// just gate on it) need the underlying ratio, not just the bool.
    pub fn evaluate(&mut self, frame: &RgbImage) -> Result<MotionReading> {
        let mat = rgb_image_to_bgr_mat(frame)?;

        let mut fgmask = Mat::default();

        self.subtractor
            .apply(&mat, &mut fgmask, -1.0)
            .context("background subtraction apply failed")?;

        #[allow(
            clippy::cast_precision_loss,
            clippy::arithmetic_side_effects,
            reason = "camera frame dimensions never approach u32::MAX / f32's 24-bit exact-integer range"
        )]
        let total_pixels = (frame.width() * frame.height()) as f32;
        let changed_ratio = changed_ratio(&fgmask, total_pixels)?;
        Ok(MotionReading {
            changed_ratio,
            tripped: changed_ratio > self.threshold,
        })
    }
}

/// Fraction of `total_pixels` marked foreground in `mask`.
fn changed_ratio(mask: &(impl MatTraitConst + ToInputArray), total_pixels: f32) -> Result<f32> {
    if total_pixels <= 0.0 {
        return Ok(0.0);
    }

    let nonzero = opencv::core::count_non_zero(mask).context("count_non_zero failed")?;

    #[allow(
        clippy::cast_precision_loss,
        reason = "nonzero pixel count never approaches f32's 24-bit exact-integer range"
    )]
    let ratio = nonzero as f32 / total_pixels;
    Ok(ratio)
}

#[cfg(test)]
mod tests {
    //! Unit tests for `MotionGate` and `changed_ratio` against a real `OpenCV`
    //! MOG2 background subtractor (no camera required). Tests that construct a
    //! real `MotionGate` acquire `detect::MODEL_TEST_LOCK` for their full
    //! duration, same as every other real-OpenCV/ONNX-Runtime-backed test in
    //! this crate, since running such tests concurrently reproduces a
    //! heap-corruption abort (see ADR 6).
    #![allow(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        clippy::missing_panics_doc,
        reason = "test assertions favor unwrap/indexing for clarity; panics here fail the test, which is the intended behavior"
    )]

    use image::Rgb;

    use super::*;
    use crate::detect;

    fn solid_frame(width: u32, height: u32, color: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(width, height, Rgb(color))
    }

    #[test]
    fn repeated_identical_frames_do_not_trip() {
        let _guard = detect::MODEL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let mut gate = MotionGate::new(0.01).unwrap();
        let frame = solid_frame(64, 64, [50, 50, 50]);

        // MOG2 needs a few frames to establish its background model before
        // its foreground mask stabilizes near zero for an unchanging scene.
        for _ in 0..5 {
            gate.evaluate(&frame).unwrap();
        }

        let reading = gate.evaluate(&frame).unwrap();

        assert!(!reading.tripped, "changed_ratio={}", reading.changed_ratio);
    }

    #[test]
    fn a_large_changed_region_trips_the_gate() {
        let _guard = detect::MODEL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let mut gate = MotionGate::new(0.01).unwrap();
        let background = solid_frame(64, 64, [50, 50, 50]);

        for _ in 0..10 {
            gate.evaluate(&background).unwrap();
        }

        let mut changed = background.clone();

        for y in 0..32 {
            for x in 0..32 {
                changed.put_pixel(x, y, Rgb([250, 250, 250]));
            }
        }

        let reading = gate.evaluate(&changed).unwrap();

        assert!(reading.tripped, "changed_ratio={}", reading.changed_ratio);
        assert!(reading.changed_ratio > 0.01);
    }

    #[test]
    fn changed_ratio_returns_zero_for_non_positive_total_pixels() {
        let mask = Mat::default();
        let ratio = changed_ratio(&mask, 0.0).unwrap();
        assert!((ratio - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn changed_ratio_errors_on_a_multi_channel_mask() {
        // `count_non_zero` requires a single-channel array (OpenCV asserts
        // `cn == 1`); a 3-channel mask reaches that assertion and surfaces as
        // an `Err`, unlike `MotionGate::new`/`evaluate`'s other fallible
        // calls, which tolerate every input this crate can construct.
        let mask = Mat::new_rows_cols_with_default(
            4,
            4,
            opencv::core::CV_8UC3,
            opencv::core::Scalar::all(0.0),
        )
        .unwrap();

        let result = changed_ratio(&mask, 16.0);

        assert!(result.is_err());
    }
}
