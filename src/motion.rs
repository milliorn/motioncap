use anyhow::{Context, Result};
use image::RgbImage;
// `Mat` ("matrix") is OpenCV's own core image/array type, not a Rust or std
// convention: https://docs.opencv.org/4.x/d3/d63/classcv_1_1Mat.html
use opencv::core::{Mat, MatTraitConst, ToInputArray};
use opencv::video::{BackgroundSubtractorTrait, create_background_subtractor_mog2};

use crate::opencv_utils::rgb_image_to_bgr_mat;

/// Number of recent frames MOG2 uses to build its rolling background model.
const MOG2_HISTORY: i32 = 500;

/// MOG2's Mahalanobis-distance-squared threshold for classifying a pixel as
/// foreground (changed) vs. background, in the background subtractor's own
/// units. `OpenCV`'s documented default.
const MOG2_VARIANCE_THRESHOLD: f64 = 16.0;

/// Sentinel passed as MOG2's `apply` learning-rate argument to mean "choose
/// an automatic rate," per `OpenCV`'s own convention for this parameter
/// (any negative value means automatic).
const MOG2_AUTO_LEARNING_RATE: f64 = -1.0;

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
        let subtractor =
            create_background_subtractor_mog2(MOG2_HISTORY, MOG2_VARIANCE_THRESHOLD, true)
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
            .apply(&mat, &mut fgmask, MOG2_AUTO_LEARNING_RATE)
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

    /// Motion-gate threshold used throughout this module's tests; low enough
    /// that the large synthetic changed-region test reliably trips it
    /// without also tripping on MOG2's own settling noise.
    const TEST_MOTION_THRESHOLD: f32 = 0.01;

    /// Frame dimension for this module's synthetic test frames; small enough
    /// to keep MOG2 warm-up fast, large enough to contain a meaningfully
    /// sized changed region.
    const TEST_FRAME_DIM: u32 = 64;

    /// A neutral background fill color for synthetic test frames.
    const BACKGROUND_COLOR: [u8; 3] = [50, 50, 50];

    /// Number of frames fed through MOG2 before asserting on a *stable*
    /// scene, giving its background model time to settle.
    const STABLE_SCENE_WARMUP_FRAMES: u32 = 5;

    /// Number of frames fed through MOG2 before introducing a changed
    /// region, giving its background model time to settle on the
    /// pre-change scene specifically (a higher count than
    /// `STABLE_SCENE_WARMUP_FRAMES` since this test's assertion is more
    /// sensitive to an unsettled background producing spurious foreground).
    const CHANGED_REGION_WARMUP_FRAMES: u32 = 10;

    /// Side length of the deliberately-changed square region within the
    /// `TEST_FRAME_DIM`-sized test frame.
    const CHANGED_REGION_SIZE: u32 = 32;

    /// A fill color for the changed region, chosen to contrast sharply
    /// against `BACKGROUND_COLOR` so MOG2 reliably classifies it as
    /// foreground.
    const CHANGED_REGION_COLOR: [u8; 3] = [250, 250, 250];

    /// Row/column count for the multi-channel-mask test's `Mat`; its exact
    /// value doesn't matter since the test expects `count_non_zero` to error
    /// before any pixel is actually inspected.
    const MULTI_CHANNEL_MASK_DIM: i32 = 4;

    /// An arbitrary nonzero `total_pixels` value for the multi-channel-mask
    /// test; irrelevant to the outcome since `count_non_zero` is expected to
    /// error before this divisor is ever used.
    const UNUSED_TOTAL_PIXELS: f32 = 16.0;

    fn solid_frame(width: u32, height: u32, color: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(width, height, Rgb(color))
    }

    #[test]
    fn repeated_identical_frames_do_not_trip() {
        let _guard = detect::MODEL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        let mut gate = MotionGate::new(TEST_MOTION_THRESHOLD).unwrap();
        let frame = solid_frame(TEST_FRAME_DIM, TEST_FRAME_DIM, BACKGROUND_COLOR);

        // MOG2 needs a few frames to establish its background model before
        // its foreground mask stabilizes near zero for an unchanging scene.
        for _ in 0..STABLE_SCENE_WARMUP_FRAMES {
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

        let mut gate = MotionGate::new(TEST_MOTION_THRESHOLD).unwrap();
        let background = solid_frame(TEST_FRAME_DIM, TEST_FRAME_DIM, BACKGROUND_COLOR);

        for _ in 0..CHANGED_REGION_WARMUP_FRAMES {
            gate.evaluate(&background).unwrap();
        }

        let mut changed = background.clone();

        for y in 0..CHANGED_REGION_SIZE {
            for x in 0..CHANGED_REGION_SIZE {
                changed.put_pixel(x, y, Rgb(CHANGED_REGION_COLOR));
            }
        }

        let reading = gate.evaluate(&changed).unwrap();

        assert!(reading.tripped, "changed_ratio={}", reading.changed_ratio);
        assert!(reading.changed_ratio > TEST_MOTION_THRESHOLD);
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
            MULTI_CHANNEL_MASK_DIM,
            MULTI_CHANNEL_MASK_DIM,
            opencv::core::CV_8UC3,
            opencv::core::Scalar::all(0.0),
        )
        .unwrap();

        let result = changed_ratio(&mask, UNUSED_TOTAL_PIXELS);

        assert!(result.is_err());
    }
}
