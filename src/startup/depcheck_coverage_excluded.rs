//! Holds `check_dependencies`, the one function in `depcheck` that cannot be
//! unit-tested: its `check_ffmpeg()?` call has an error-propagation branch
//! untestable for the same reason documented on `check_ffmpeg` itself in
//! `depcheck.rs` (faking `PATH` needs `unsafe`, denied by this crate).
//! `cargo-llvm-cov` attributes that branch to this call site regardless of
//! which function it lives in, so only moving the whole function out (not
//! `check_ffmpeg`, `check_ffmpeg_probe`, or `check_model_file`, which stay in
//! `depcheck.rs` fully tested) clears it. See
//! `docs/adr/0006-coverage-exclusions.md`.

use anyhow::Result;

use super::depcheck::{check_ffmpeg, check_model_file};
use crate::config::Config;

/// Verifies hard runtime requirements are present and exits the process with a
/// clear, actionable message if not. Per ADR 5, this never attempts to install
/// anything itself, only detects and reports.
pub fn check_dependencies(config: &Config) -> Result<()> {
    check_ffmpeg()?;
    check_model_file(&config.model_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        reason = "test assertions favor unwrap for clarity; panics here fail the test, which is the intended behavior"
    )]

    use clap::Parser as _;
    use tempfile::NamedTempFile;

    use super::*;

    #[test]
    fn check_dependencies_errors_when_model_file_is_missing() {
        let config =
            Config::try_parse_from(["motioncap", "--model-path", "/nonexistent/path/model.onnx"])
                .unwrap();

        let err = check_dependencies(&config).unwrap_err();
        assert!(err.to_string().contains("ONNX model file not found"));
    }

    #[test]
    fn check_dependencies_ok_when_ffmpeg_and_model_present() {
        let file = NamedTempFile::new().unwrap();
        let config =
            Config::try_parse_from(["motioncap", "--model-path", &file.path().to_string_lossy()])
                .unwrap();

        assert!(check_dependencies(&config).is_ok());
    }
}
