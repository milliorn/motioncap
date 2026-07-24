use std::path::{Path, PathBuf};

use clap::Parser;

/// motioncap: webcam-based security motion capture
#[derive(Parser, Debug)]
#[command(name = "motioncap", version)]
pub struct Config {
    /// Directory recordings are written to (organized as <date>/<time>_<classes>.mp4)
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
    #[arg(long, default_value_t = 10)]
    pub pre_buffer_secs: u32,

    /// Seconds to keep recording after the last trigger before closing the clip
    #[arg(long, default_value_t = 15)]
    pub post_buffer_secs: u32,

    /// Minimum confidence (0.0-1.0) for a YOLO detection to confirm a living-thing event
    #[arg(long, default_value_t = 0.5)]
    pub detection_confidence: f32,

    /// Minimum changed-pixel ratio (0.0-1.0) for the background-subtraction gate to trip
    #[arg(long, default_value_t = 0.01)]
    pub motion_threshold: f32,

    /// Show a live preview window with the raw camera feed (off by default)
    #[arg(long, default_value_t = false)]
    pub preview: bool,
}

impl Config {
    pub fn parse_args() -> Self {
        Self::parse()
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }
}
