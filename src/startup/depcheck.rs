use std::path::Path;
use std::process::Command;

use anyhow::{bail, Result};

use crate::config::Config;

/// Verifies hard runtime requirements are present and exits the process with a
/// clear, actionable message if not. Per ADR 5, this never attempts to install
/// anything itself — only detects and reports.
pub fn check_dependencies(config: &Config) -> Result<()> {
    check_ffmpeg()?;
    check_model_file(&config.model_path)?;
    Ok(())
}

fn check_ffmpeg() -> Result<()> {
    let found = Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false);

    if !found {
        bail!(
            "ffmpeg was not found on this system, but motioncap requires it for video encoding.\n\
             Install it with one of:\n  \
             Debian/Ubuntu:  sudo apt install ffmpeg\n  \
             Arch/CachyOS:   sudo pacman -S ffmpeg\n  \
             Fedora:         sudo dnf install ffmpeg\n  \
             macOS:          brew install ffmpeg\n\
             Then re-run motioncap."
        );
    }
    Ok(())
}

fn check_model_file(path: &Path) -> Result<()> {
    if !path.is_file() {
        bail!(
            "ONNX model file not found at {}.\n\
             Pass --model-path pointing to a YOLO model exported to ONNX format\n\
             (e.g. YOLOv8n, exported once via the `ultralytics` Python package).",
            path.display()
        );
    }
    Ok(())
}
