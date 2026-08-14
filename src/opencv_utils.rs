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

#[cfg(test)]
mod tests {
    //! Unit tests for the RGB-to-BGR `Mat` conversion against a real `OpenCV` `Mat`.
    #![allow(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        clippy::missing_panics_doc,
        reason = "test assertions favor unwrap/indexing for clarity; panics here fail the test, which is the intended behavior"
    )]

    use image::Rgb;

    use super::*;

    #[test]
    fn converts_rgb_pixels_to_bgr_channel_order() {
        let mut image = RgbImage::new(2, 2);
        image.put_pixel(0, 0, Rgb([10, 20, 30]));
        image.put_pixel(1, 0, Rgb([40, 50, 60]));
        image.put_pixel(0, 1, Rgb([70, 80, 90]));
        image.put_pixel(1, 1, Rgb([100, 110, 120]));

        let mat = rgb_image_to_bgr_mat(&image).unwrap();

        let px = *mat.at_2d::<Vec3b>(0, 0).unwrap();
        assert_eq!(px.0, [30, 20, 10]);

        let px = *mat.at_2d::<Vec3b>(0, 1).unwrap();
        assert_eq!(px.0, [60, 50, 40]);

        let px = *mat.at_2d::<Vec3b>(1, 0).unwrap();
        assert_eq!(px.0, [90, 80, 70]);

        let px = *mat.at_2d::<Vec3b>(1, 1).unwrap();
        assert_eq!(px.0, [120, 110, 100]);
    }

    #[test]
    fn produces_mat_with_matching_dimensions() {
        let image = RgbImage::new(5, 3);
        let mat = rgb_image_to_bgr_mat(&image).unwrap();

        assert_eq!(mat.rows(), 3);
        assert_eq!(mat.cols(), 5);
    }
}
