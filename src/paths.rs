use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Local};

/// Builds the `<output_dir>/<YYYY-MM-DD>/<HH-MM-SS>_<class1>_<class2>.mp4` path
/// for a recorded clip (ADR 4), creating the day's folder if needed. `classes`
/// is deduplicated and sorted alphabetically so multi-subject clips get a
/// stable, predictable filename regardless of detection order.
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

    let mut filename = started_at.format("%H-%M-%S").to_string();

    for class in &sorted_classes {
        filename.push('_');
        filename.push_str(class);
    }

    filename.push_str(".mp4");

    Ok(day_dir.join(filename))
}

/// The sidecar JSON path for a given clip path, e.g. `foo.mp4` -> `foo.json`.
pub fn sidecar_path(clip_path: &Path) -> PathBuf {
    clip_path.with_extension("json")
}
