//! motioncap: webcam-based security motion capture. See the project guidance
//! file and `docs/adr/` for architecture and design-decision context.

/// Entry point. Delegates immediately to `app::run`, which opens the real
/// camera/audio devices and spawns long-lived worker threads; this file has
/// no logic of its own for an automated test to exercise.
fn main() -> anyhow::Result<()> {
    motioncap::app::run()
}
