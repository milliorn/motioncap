/// Hard runtime dependency detection (ffmpeg, ONNX model file).
mod depcheck;
/// `check_dependencies`, the one function in `depcheck` that cannot be
/// unit-tested (see that module's doc comment).
mod depcheck_coverage_excluded;

/// Lives in `depcheck_coverage_excluded`, not `depcheck`; see that module's
/// doc comment.
pub use depcheck_coverage_excluded::check_dependencies;
