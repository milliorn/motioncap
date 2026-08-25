use std::path::{Path, PathBuf};

use clap::Parser;

/// Default `--pre-buffer-secs`. Named (not just a bare `default_value_t`
/// literal) so `config.rs`'s own tests can assert against the same value
/// `Config` actually defaults to, rather than a second, independently
/// hardcoded copy of it.
const DEFAULT_PRE_BUFFER_SECS: u32 = 10;

/// Default `--post-buffer-secs`. See `DEFAULT_PRE_BUFFER_SECS`.
const DEFAULT_POST_BUFFER_SECS: u32 = 15;

/// Default `--detection-confidence`. See `DEFAULT_PRE_BUFFER_SECS`.
const DEFAULT_DETECTION_CONFIDENCE: f32 = 0.3;

/// Default `--motion-threshold`. See `DEFAULT_PRE_BUFFER_SECS`.
const DEFAULT_MOTION_THRESHOLD: f32 = 0.01;

/// motioncap: webcam-based security motion capture
#[derive(Parser, Debug)]
#[command(name = "motioncap", version)]
pub struct Config {
    /// Directory recordings are written to (organized as `<date>/<date>_<time>_<classes>.mp4`)
    #[arg(long, default_value = "./recordings")]
    pub output_dir: PathBuf,

    /// Path to the YOLO ONNX model file
    #[arg(long, default_value = "./models/yolov8n.onnx")]
    pub model_path: PathBuf,

    /// Camera device path (e.g. /dev/video0). Auto-detected if not set.
    #[arg(long)]
    pub camera_device: Option<PathBuf>,

    /// Force CPU-only inference even if a GPU accelerator is available
    #[arg(long, default_value_t = false)]
    pub force_cpu: bool,

    /// Seconds of video/audio to keep buffered before a trigger
    #[arg(long, default_value_t = DEFAULT_PRE_BUFFER_SECS)]
    pub pre_buffer_secs: u32,

    /// Seconds to keep recording after the last trigger before closing the clip
    #[arg(long, default_value_t = DEFAULT_POST_BUFFER_SECS)]
    pub post_buffer_secs: u32,

    /// Minimum confidence (0.0-1.0) for a YOLO detection to confirm a living-thing event
    #[arg(long, default_value_t = DEFAULT_DETECTION_CONFIDENCE)]
    pub detection_confidence: f32,

    /// Minimum changed-pixel ratio (0.0-1.0) for the background-subtraction gate to trip
    #[arg(long, default_value_t = DEFAULT_MOTION_THRESHOLD)]
    pub motion_threshold: f32,

    /// Show a live preview window with the raw camera feed (off by default)
    #[arg(long, default_value_t = false)]
    pub preview: bool,
}

impl Config {
    /// The configured YOLO ONNX model path.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }
}

/// Parses `Config` from the process's command-line arguments. Calls clap's
/// `Config::parse()`, which reads the real process's `std::env::args()`; a
/// test can't safely override argv for the running test binary the way
/// `Config::try_parse_from` lets tests supply synthetic argv, so this
/// function itself is not covered by an automated test.
pub fn parse_args() -> Config {
    Config::parse()
}

#[cfg(test)]
mod tests {
    //! Unit tests for `Config`'s CLI flag defaults and overrides.
    #![allow(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        clippy::missing_panics_doc,
        reason = "test assertions favor unwrap/indexing for clarity; panics here fail the test, which is the intended behavior"
    )]

    use super::*;

    /// Overridden `--pre-buffer-secs` value for `flags_are_overridable`,
    /// distinct from `DEFAULT_PRE_BUFFER_SECS` so the test actually proves
    /// the flag changed the value rather than coincidentally matching it.
    const OVERRIDE_PRE_BUFFER_SECS: u32 = 5;

    /// Overridden `--post-buffer-secs` value. See `OVERRIDE_PRE_BUFFER_SECS`.
    const OVERRIDE_POST_BUFFER_SECS: u32 = 20;

    /// Overridden `--detection-confidence` value. See `OVERRIDE_PRE_BUFFER_SECS`.
    const OVERRIDE_DETECTION_CONFIDENCE: f32 = 0.7;

    /// Overridden `--motion-threshold` value. See `OVERRIDE_PRE_BUFFER_SECS`.
    const OVERRIDE_MOTION_THRESHOLD: f32 = 0.05;

    #[test]
    fn defaults_match_documented_values() {
        let config = Config::try_parse_from(["motioncap"]).unwrap();

        assert_eq!(config.output_dir, PathBuf::from("./recordings"));
        assert_eq!(config.model_path, PathBuf::from("./models/yolov8n.onnx"));
        assert_eq!(config.camera_device, None);
        assert!(!config.force_cpu);
        assert_eq!(config.pre_buffer_secs, DEFAULT_PRE_BUFFER_SECS);
        assert_eq!(config.post_buffer_secs, DEFAULT_POST_BUFFER_SECS);
        assert!((config.detection_confidence - DEFAULT_DETECTION_CONFIDENCE).abs() < f32::EPSILON);
        assert!((config.motion_threshold - DEFAULT_MOTION_THRESHOLD).abs() < f32::EPSILON);
        assert!(!config.preview);
    }

    #[test]
    fn flags_are_overridable() {
        let config = Config::try_parse_from([
            "motioncap",
            "--output-dir",
            "/tmp/out",
            "--model-path",
            "/tmp/model.onnx",
            "--camera-device",
            "/dev/video1",
            "--force-cpu",
            "--pre-buffer-secs",
            &OVERRIDE_PRE_BUFFER_SECS.to_string(),
            "--post-buffer-secs",
            &OVERRIDE_POST_BUFFER_SECS.to_string(),
            "--detection-confidence",
            &OVERRIDE_DETECTION_CONFIDENCE.to_string(),
            "--motion-threshold",
            &OVERRIDE_MOTION_THRESHOLD.to_string(),
            "--preview",
        ])
        .unwrap();

        assert_eq!(config.output_dir, PathBuf::from("/tmp/out"));
        assert_eq!(config.model_path, PathBuf::from("/tmp/model.onnx"));
        assert_eq!(config.camera_device, Some(PathBuf::from("/dev/video1")));
        assert!(config.force_cpu);
        assert_eq!(config.pre_buffer_secs, OVERRIDE_PRE_BUFFER_SECS);
        assert_eq!(config.post_buffer_secs, OVERRIDE_POST_BUFFER_SECS);
        assert!((config.detection_confidence - OVERRIDE_DETECTION_CONFIDENCE).abs() < f32::EPSILON);
        assert!((config.motion_threshold - OVERRIDE_MOTION_THRESHOLD).abs() < f32::EPSILON);
        assert!(config.preview);
    }

    #[test]
    fn invalid_numeric_flag_is_rejected() {
        let result = Config::try_parse_from(["motioncap", "--pre-buffer-secs", "not-a-number"]);
        assert!(result.is_err());
    }

    #[test]
    fn model_path_accessor_returns_configured_path() {
        let config = Config::try_parse_from(["motioncap", "--model-path", "/x/y.onnx"]).unwrap();
        assert_eq!(config.model_path(), Path::new("/x/y.onnx"));
    }
}
