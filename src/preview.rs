use anyhow::{Context, Result};
use image::RgbImage;
use opencv::core::{Mat, Vec3b};
use opencv::highgui;

const WINDOW_NAME: &str = "motioncap preview";

/// Opt-in live preview window (ADR: off by default). Entirely decoupled from
/// the recording pipeline -- it only ever displays frames, never affects
/// whether or how a clip is recorded.
pub struct PreviewWindow;

impl PreviewWindow {
    pub fn open() -> Result<Self> {
        highgui::named_window(WINDOW_NAME, highgui::WINDOW_AUTOSIZE)
            .context("failed to open preview window")?;
        
        Ok(Self)
    }

    /// Displays one frame. Must be called periodically from the main loop for
    /// the window to pump its event queue and stay responsive.
    pub fn show(&mut self, frame: &RgbImage) -> Result<()> {
        let mat = rgb_image_to_mat(frame)?;

        highgui::imshow(WINDOW_NAME, &mat).context("failed to display preview frame")?;
        highgui::wait_key(1).context("failed to pump preview window events")?;

        Ok(())
    }
}

impl Drop for PreviewWindow {
    fn drop(&mut self) {
        let _ = highgui::destroy_window(WINDOW_NAME);
    }
}

fn rgb_image_to_mat(image: &RgbImage) -> Result<Mat> {
    let (width, height) = image.dimensions();

    let bgr: Vec<Vec3b> = image
        .pixels()
        .map(|p| Vec3b::from([p[2], p[1], p[0]]))
        .collect();

    let borrowed = Mat::new_rows_cols_with_data(height as i32, width as i32, &bgr)
        .context("failed to build Mat from frame")?;

    opencv::core::MatTraitConst::try_clone(&borrowed).context("failed to clone frame Mat")
}
