use std::path::Path;
use std::process::Command;

use anyhow::{Result, bail};

use crate::config::Config;

/// Verifies hard runtime requirements are present and exits the process with a
/// clear, actionable message if not. Per ADR 5, this never attempts to install
/// anything itself — only detects and reports.
pub fn check_dependencies(config: &Config) -> Result<()> {
    check_dependencies_with_ffmpeg_binary(config, "ffmpeg")
}

/// Same as `check_dependencies`, parameterized on the ffmpeg binary name so
/// tests can exercise both `?`-propagation arms (ffmpeg missing, model
/// missing) by probing a name that isn't on `PATH`, without needing to fake
/// `PATH` itself (this crate denies `unsafe_code`, which `std::env::set_var`
/// requires on current Rust).
fn check_dependencies_with_ffmpeg_binary(config: &Config, ffmpeg_binary: &str) -> Result<()> {
    check_ffmpeg_binary(ffmpeg_binary)?;
    check_model_file(&config.model_path)?;
    Ok(())
}

/// Probes for a `-version`-capable binary by name, exiting with install
/// instructions if not found. Split out from `check_ffmpeg` so the not-found
/// path can be exercised in tests by probing a name that isn't on `PATH`,
/// without needing to actually uninstall ffmpeg.
fn check_ffmpeg_binary(binary: &str) -> Result<()> {
    let found = Command::new(binary)
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success());

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

/// Checks that the configured ONNX model file exists.
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

#[cfg(test)]
mod tests {
    //! Unit tests for startup dependency checks. `check_ffmpeg_binary("ffmpeg")`
    //! is tested against the real `PATH` (ffmpeg is a hard runtime dependency
    //! of this crate per ADR 5, so it's expected to be present wherever these
    //! tests run) rather than mocked, since mocking a subprocess probe would
    //! test nothing real about whether ffmpeg is actually reachable. The
    //! not-found path is instead exercised by probing a binary name that
    //! doesn't exist on `PATH`.
    #![allow(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        clippy::missing_panics_doc,
        reason = "test assertions favor unwrap/indexing for clarity; panics here fail the test, which is the intended behavior"
    )]

    use tempfile::NamedTempFile;

    use super::*;

    #[test]
    fn check_ffmpeg_binary_succeeds_when_present_on_path() {
        assert!(check_ffmpeg_binary("ffmpeg").is_ok());
    }

    #[test]
    fn check_ffmpeg_binary_errors_with_helpful_message_when_not_found() {
        let err = check_ffmpeg_binary("definitely-not-a-real-binary-motioncap-test").unwrap_err();
        let message = err.to_string();
        assert!(message.contains("ffmpeg was not found on this system"));
        assert!(message.contains("sudo apt install ffmpeg"));
    }

    #[test]
    fn check_model_file_ok_when_file_exists() {
        let file = NamedTempFile::new().unwrap();
        assert!(check_model_file(file.path()).is_ok());
    }

    #[test]
    fn check_dependencies_errors_when_ffmpeg_binary_is_missing() {
        use clap::Parser as _;

        let file = NamedTempFile::new().unwrap();
        let config =
            Config::try_parse_from(["motioncap", "--model-path", &file.path().to_string_lossy()])
                .unwrap();

        let err = check_dependencies_with_ffmpeg_binary(
            &config,
            "definitely-not-a-real-binary-motioncap-test",
        )
        .unwrap_err();
        assert!(err.to_string().contains("ffmpeg was not found"));
    }

    #[test]
    fn check_dependencies_errors_when_model_file_is_missing() {
        use clap::Parser as _;

        let config =
            Config::try_parse_from(["motioncap", "--model-path", "/nonexistent/path/model.onnx"])
                .unwrap();

        let err = check_dependencies(&config).unwrap_err();
        assert!(err.to_string().contains("ONNX model file not found"));
    }

    #[test]
    fn check_dependencies_ok_when_ffmpeg_and_model_present() {
        use clap::Parser as _;

        let file = NamedTempFile::new().unwrap();
        let config =
            Config::try_parse_from(["motioncap", "--model-path", &file.path().to_string_lossy()])
                .unwrap();

        assert!(check_dependencies(&config).is_ok());
    }

    #[test]
    fn check_model_file_errors_with_helpful_message_when_missing() {
        let missing = Path::new("/nonexistent/path/model.onnx");
        let err = check_model_file(missing).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("ONNX model file not found"));
        assert!(message.contains("/nonexistent/path/model.onnx"));
    }
}
