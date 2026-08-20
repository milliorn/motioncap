/// Hard runtime dependency detection (ffmpeg, ONNX model file).
mod depcheck;
/// `check_dependencies`, isolated because its `check_ffmpeg()?` call site has
/// an error-propagation branch untestable without faking `PATH`. See
/// `docs/adr/0006-coverage-exclusions.md`.
mod depcheck_coverage_excluded;

pub use depcheck_coverage_excluded::check_dependencies;
