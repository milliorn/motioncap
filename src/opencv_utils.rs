use anyhow::{Context, Result};
use image::RgbImage;
use opencv::core::{Mat, MatTraitConst, Vec3b};

/// Converts an RGB image into an owned `OpenCV` BGR `Mat`, for use with any
/// `OpenCV` API (background subtraction, highgui display, ...) that expects
/// BGR pixel order.
pub fn rgb_image_to_bgr_mat(image: &RgbImage) -> Result<Mat> {
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
