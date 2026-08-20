//! Holds `check_dependencies`, whose `check_ffmpeg()?` call site has an
//! error-propagation branch that only reachable if `ffmpeg` is genuinely
//! absent from `PATH`. That's the same untestable condition documented on
//! `check_ffmpeg` itself in `depcheck.rs` (this crate denies `unsafe_code`,
//! and faking `PATH` from within a test requires `std::env::set_var`, which
//! needs `unsafe` on current Rust) - just visible one call frame up. Since
//! `cargo-llvm-cov` attributes an uncovered `?` region to its own source
//! line regardless of which function the fallible call lives in, leaving
//! `check_dependencies` in `depcheck.rs` would hold that whole file below
//! 100% for a branch that isn't actually exercisable there either. Moving
//! only this thin orchestration function out (not `check_ffmpeg`,
//! `check_ffmpeg_probe`, or `check_model_file`, which are all fully tested
//! and stay in `depcheck.rs`) mirrors the `RecordingEvent::start`/`finish`
//! split in `recorder_coverage_excluded.rs`. See
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
