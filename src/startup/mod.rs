/// Hard runtime dependency detection (ffmpeg, ONNX model file).
mod depcheck;

pub use depcheck::check_dependencies;
