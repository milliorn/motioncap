// coverage: excluded (whole file); every function here drives OpenCV's
// highgui, which requires a real X11/Wayland display; there is no headless
// fake for a GUI window, so this is untestable in CI or a headless dev
// session. See docs/adr/0006-coverage-exclusions.md.

use std::cell::Cell;

use anyhow::{Context, Result};
use image::RgbImage;
use opencv::highgui;
use opencv::prelude::MatTraitConst;

use crate::opencv_utils::rgb_image_to_bgr_mat;

/// Title of the `OpenCV` highgui preview window.
const WINDOW_NAME: &str = "motioncap preview";

/// Factor the window is scaled up from the camera's native capture
/// resolution. Webcams commonly capture at low resolutions (e.g.
/// 320x240), which renders unreadably small with a 1:1 `WINDOW_AUTOSIZE`
/// window on a modern display.
const PREVIEW_SCALE: i32 = 2;

/// Minimum wait, in milliseconds, `highgui::wait_key` blocks for a keypress
/// before returning; pumps the window's event queue without waiting for
/// human input to be present, since nothing here reads keyboard state.
const WAIT_KEY_MILLIS: i32 = 1;

/// Opt-in live preview window (ADR: off by default). Entirely decoupled from
/// the recording pipeline: it only ever displays frames, never affects
/// whether or how a clip is recorded.
pub struct PreviewWindow {
    /// Tracks whether the window has been sized to the first displayed
    /// frame yet; the frame's native resolution isn't known at `open()`.
    resized: Cell<bool>,
}

impl PreviewWindow {
    /// Opens the highgui preview window.
    pub fn open() -> Result<Self> {
        highgui::named_window(WINDOW_NAME, highgui::WINDOW_NORMAL)
            .context("failed to open preview window")?;

        Ok(Self {
            resized: Cell::new(false),
        })
    }

    /// Displays one frame. Must be called periodically from the main loop for
    /// the window to pump its event queue and stay responsive.
    pub fn show(&self, frame: &RgbImage) -> Result<()> {
        let mat = rgb_image_to_bgr_mat(frame)?;

        if !self.resized.get() {
            highgui::resize_window(
                WINDOW_NAME,
                mat.cols().saturating_mul(PREVIEW_SCALE),
                mat.rows().saturating_mul(PREVIEW_SCALE),
            )
            .context("failed to resize preview window")?;
            self.resized.set(true);
        }

        highgui::imshow(WINDOW_NAME, &mat).context("failed to display preview frame")?;
        highgui::wait_key(WAIT_KEY_MILLIS).context("failed to pump preview window events")?;

        Ok(())
    }
}

impl Drop for PreviewWindow {
    fn drop(&mut self) {
        let _ = highgui::destroy_window(WINDOW_NAME);
    }
}
