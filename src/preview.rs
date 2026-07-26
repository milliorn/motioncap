use anyhow::{Context, Result};
use image::RgbImage;
use opencv::highgui;

use crate::opencv_utils::rgb_image_to_bgr_mat;

/// Title of the `OpenCV` highgui preview window.
const WINDOW_NAME: &str = "motioncap preview";

/// Opt-in live preview window (ADR: off by default). Entirely decoupled from
/// the recording pipeline -- it only ever displays frames, never affects
/// whether or how a clip is recorded.
pub struct PreviewWindow;

impl PreviewWindow {
    /// Opens the highgui preview window.
    pub fn open() -> Result<Self> {
        highgui::named_window(WINDOW_NAME, highgui::WINDOW_AUTOSIZE)
            .context("failed to open preview window")?;

        Ok(Self)
    }

    /// Displays one frame. Must be called periodically from the main loop for
    /// the window to pump its event queue and stay responsive.
    #[allow(
        clippy::unused_self,
        reason = "self ties display to the open window's RAII handle even though the call is stateless"
    )]
    pub fn show(&self, frame: &RgbImage) -> Result<()> {
        let mat = rgb_image_to_bgr_mat(frame)?;

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
