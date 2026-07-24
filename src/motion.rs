use anyhow::{Context, Result};
use image::RgbImage;
use opencv::core::{Mat, MatTraitConst, ToInputArray, Vec3b};
use opencv::video::{create_background_subtractor_mog2, BackgroundSubtractorTrait};

/// Background-subtraction motion gate (ADR 2). Its only job is deciding
/// whether whole-frame motion exceeded the configured threshold; recording
/// itself is only ever triggered by a subsequent confirmed YOLO
/// classification, never by this gate alone.
pub struct MotionGate {
    subtractor: opencv::core::Ptr<opencv::video::BackgroundSubtractorMOG2>,
    threshold: f32,
}

impl MotionGate {
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

        let total_pixels = (frame.width() * frame.height()) as f32;
        Ok(changed_ratio(&fgmask, total_pixels)? > self.threshold)
    }
}

fn changed_ratio(mask: &(impl MatTraitConst + ToInputArray), total_pixels: f32) -> Result<f32> {
    if total_pixels <= 0.0 {
        return Ok(0.0);
    }
    let nonzero = opencv::core::count_non_zero(mask).context("count_non_zero failed")?;
    Ok(nonzero as f32 / total_pixels)
}

fn rgb_image_to_mat(image: &RgbImage) -> Result<Mat> {
    let (width, height) = image.dimensions();
    let bgr: Vec<Vec3b> = image
        .pixels()
        .map(|p| Vec3b::from([p[2], p[1], p[0]]))
        .collect();
    let borrowed = Mat::new_rows_cols_with_data(height as i32, width as i32, &bgr)
        .context("failed to build Mat from frame")?;
    borrowed.try_clone().context("failed to clone frame Mat")
}
