use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Local};

/// Builds the `<output_dir>/<YYYY-MM-DD>/<YYYY-MM-DD_HH-MM-SS>_<class1>_<class2>.mp4`
/// path for a recorded clip (ADR 4), creating the day's folder if needed.
///
/// The full date is repeated in the filename (not just the containing
/// folder) so that filenames sort chronologically by name alone (e.g. in a
/// flat listing across multiple days' folders), rather than only within a
/// single day's folder. `classes` is deduplicated and sorted alphabetically
/// so multi-subject clips get a stable, predictable filename regardless of
/// detection order.
///
/// # Errors
///
/// Returns an error if the day's output directory can't be created.
pub fn clip_path(
    output_dir: &Path,
    started_at: DateTime<Local>,
    classes: &[&str],
) -> Result<PathBuf> {
    let day_dir = output_dir.join(started_at.format("%Y-%m-%d").to_string());

    std::fs::create_dir_all(&day_dir)
        .with_context(|| format!("failed to create output directory {}", day_dir.display()))?;

    let mut sorted_classes: Vec<&str> = classes.to_vec();

    sorted_classes.sort_unstable();
    sorted_classes.dedup();

    let mut filename = started_at.format("%Y-%m-%d_%H-%M-%S").to_string();

    for class in &sorted_classes {
        filename.push('_');
        filename.push_str(class);
    }

    filename.push_str(".mp4");

    Ok(day_dir.join(filename))
}

/// The sidecar JSON path for a given clip path, e.g. `foo.mp4` -> `foo.json`.
#[must_use]
pub fn sidecar_path(clip_path: &Path) -> PathBuf {
    clip_path.with_extension("json")
}

#[cfg(test)]
mod tests {
    //! Unit tests for clip/sidecar path construction (ADR 4).
    #![allow(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        clippy::missing_panics_doc,
        reason = "test assertions favor unwrap/indexing for clarity; panics here fail the test, which is the intended behavior"
    )]

    use chrono::TimeZone;
    use tempfile::tempdir;

    use super::*;

    fn fixed_time() -> DateTime<Local> {
        Local.with_ymd_and_hms(2026, 3, 5, 13, 45, 30).unwrap()
    }

    #[test]
    fn clip_path_has_expected_folder_and_filename_format() {
        let dir = tempdir().unwrap();
        let path = clip_path(dir.path(), fixed_time(), &["person"]).unwrap();

        assert_eq!(
            path,
            dir.path()
                .join("2026-03-05")
                .join("2026-03-05_13-45-30_person.mp4")
        );
        assert!(path.parent().unwrap().is_dir());
    }

    #[test]
    fn clip_path_dedups_and_sorts_classes_regardless_of_input_order() {
        let dir = tempdir().unwrap();
        let path = clip_path(dir.path(), fixed_time(), &["dog", "person", "dog", "cat"]).unwrap();

        let filename = path.file_name().unwrap().to_string_lossy().into_owned();
        assert_eq!(filename, "2026-03-05_13-45-30_cat_dog_person.mp4");
    }

    #[test]
    fn clip_path_creates_day_directory() {
        let dir = tempdir().unwrap();
        let day_dir = dir.path().join("2026-03-05");
        assert!(!day_dir.exists());

        clip_path(dir.path(), fixed_time(), &["person"]).unwrap();

        assert!(day_dir.is_dir());
    }

    #[test]
    fn clip_path_errors_when_output_dir_cannot_be_created() {
        let dir = tempdir().unwrap();
        let blocked = dir.path().join("blocked");
        std::fs::write(&blocked, b"not a directory").unwrap();

        let err = clip_path(&blocked, fixed_time(), &["person"]).unwrap_err();

        assert!(
            err.to_string()
                .contains("failed to create output directory")
        );
    }

    #[test]
    fn sidecar_path_swaps_extension_only() {
        let clip = Path::new("/recordings/2026-03-05/2026-03-05_13-45-30_person.mp4");
        let sidecar = sidecar_path(clip);

        assert_eq!(
            sidecar,
            Path::new("/recordings/2026-03-05/2026-03-05_13-45-30_person.json")
        );
    }
}
