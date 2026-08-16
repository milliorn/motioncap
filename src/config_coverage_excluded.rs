//! Holds `parse_args`, the one function in `config` that cannot be
//! exercised by an automated test under any circumstances: it calls clap's
//! `Config::parse()`, which reads the real process's `std::env::args()`,
//! and a test can't safely override argv for the running test binary the
//! way `Config::try_parse_from` lets tests supply synthetic argv. Split out
//! of `config.rs` (same convention as `coverage_excluded.rs` at the crate
//! root) so that file can reach genuine 100% coverage instead of being held
//! down to whatever fraction this one untestable function happens to be.
//! See `docs/adr/0006-coverage-exclusions.md`.

use clap::Parser;

use crate::config::Config;

/// Parses `Config` from the process's command-line arguments.
pub fn parse_args() -> Config {
    Config::parse()
}
