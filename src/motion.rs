use anyhow::{Context, Result};
use image::RgbImage;
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
    /// threshold -- callers that need to log/record motion activity (not
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
