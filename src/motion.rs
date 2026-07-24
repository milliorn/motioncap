use anyhow::{Context, Result};
use image::RgbImage;
use opencv::core::{Mat, MatTraitConst, ToInputArray, Vec3b};
use opencv::video::{BackgroundSubtractorTrait, create_background_subtractor_mog2};

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

    /// Feeds one frame through the background model and reports whether the
    /// changed-pixel ratio exceeded the configured threshold.
    pub fn evaluate(&mut self, frame: &RgbImage) -> Result<bool> {
        let mat = rgb_image_to_mat(frame)?;

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
        Ok(changed_ratio(&fgmask, total_pixels)? > self.threshold)
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

/// Converts an RGB image into an `OpenCV` BGR `Mat` for use with the background subtractor.
fn rgb_image_to_mat(image: &RgbImage) -> Result<Mat> {
    let (width, height) = image.dimensions();
    let bgr: Vec<Vec3b> = image
        .pixels()
        .map(|p| Vec3b::from([p[2], p[1], p[0]]))
        .collect();
    #[allow(
        clippy::cast_possible_wrap,
        reason = "camera frame dimensions never approach i32::MAX"
    )]
    let borrowed = Mat::new_rows_cols_with_data(height as i32, width as i32, &bgr)
        .context("failed to build Mat from frame")?;
    borrowed.try_clone().context("failed to clone frame Mat")
}
