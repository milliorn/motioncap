/// Audio capture via cpal.
pub mod audio;
/// `start_audio_capture`, the one function in `audio` that cannot be
/// unit-tested (see that module's doc comment).
pub mod audio_coverage_excluded;
/// Camera capture via nokhwa.
pub mod camera;
/// `start_camera_capture` and camera auto-detect, the parts of `camera` that
/// cannot be unit-tested (see that module's doc comment).
pub mod camera_coverage_excluded;
