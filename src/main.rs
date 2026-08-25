//! motioncap: webcam-based security motion capture. See the project guidance
//! file and `docs/adr/` for architecture and design-decision context.

/// Entry point. Opens the real camera/audio devices and spawns long-lived
/// worker threads, so it's not exercised by an automated test.
fn main() -> anyhow::Result<()> {
    motioncap::app::run()
}
