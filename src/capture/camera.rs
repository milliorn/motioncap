use std::path::Path;

use anyhow::{Context, Result};
use nokhwa::utils::CameraIndex;

/// Resolves a configured camera device (e.g. `/dev/video0`) to a nokhwa
/// `CameraIndex`, or `Ok(None)` if no device was pinned (the caller should
/// then auto-select one).
///
/// On Linux, nokhwa's v4l backend expects `CameraIndex::Index(N)` (matching
/// `/dev/videoN`), not a path string. `CameraIndex::String` is reserved for
/// IP cameras.
pub(super) fn resolve_pinned_camera_index(device: Option<&Path>) -> Result<Option<CameraIndex>> {
    let Some(path) = device else {
        return Ok(None);
    };

    let name = path
        .to_str()
        .context("camera device path must be valid UTF-8")?;

    let index_str = name.trim_start_matches("/dev/video");

    let index: u32 = index_str
        .parse()
        .with_context(|| format!("expected a /dev/videoN path, got {name}"))?;

    Ok(Some(CameraIndex::Index(index)))
}

#[cfg(test)]
mod tests {
    //! Unit tests for `resolve_pinned_camera_index`. The auto-detect path
    //! (`auto_detect_camera_index` in `camera_coverage_excluded.rs`) and
    //! `start_camera_capture` require a real camera device and are left
    //! untested here (see `docs/adr/0006-coverage-exclusions.md`).
    #![allow(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        clippy::missing_panics_doc,
        reason = "test assertions favor unwrap/indexing for clarity; panics here fail the test, which is the intended behavior"
    )]

    use super::*;

    #[test]
    fn resolves_dev_video_path_to_matching_index() {
        let index = resolve_pinned_camera_index(Some(Path::new("/dev/video0"))).unwrap();
        assert_eq!(index, Some(CameraIndex::Index(0)));
    }

    #[test]
    fn resolves_dev_video_path_with_multi_digit_index() {
        let index = resolve_pinned_camera_index(Some(Path::new("/dev/video12"))).unwrap();
        assert_eq!(index, Some(CameraIndex::Index(12)));
    }

    #[test]
    fn rejects_non_numeric_suffix() {
        let result = resolve_pinned_camera_index(Some(Path::new("/dev/videoX")));
        assert!(result.is_err());
    }

    #[test]
    fn rejects_path_not_matching_dev_video_pattern() {
        // trim_start_matches leaves "/dev/webcam0" fully intact (no prefix
        // match), which then fails to parse as a u32.
        let result = resolve_pinned_camera_index(Some(Path::new("/dev/webcam0")));
        assert!(result.is_err());
    }

    #[test]
    fn returns_none_when_no_device_pinned() {
        let index = resolve_pinned_camera_index(None).unwrap();
        assert_eq!(index, None);
    }

    #[test]
    fn rejects_non_utf8_path() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        let invalid = OsStr::from_bytes(&[0x66, 0x6f, 0x80, 0x6f]);
        let result = resolve_pinned_camera_index(Some(Path::new(invalid)));
        assert!(result.is_err());
    }
}
