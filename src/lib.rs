//! motioncap: webcam-based security motion capture. See the project guidance
//! file and `docs/adr/` for architecture and design-decision context.
//!
//! Most modules here are `pub(crate)`: this crate ships as a single binary
//! (see `docs/adr/0001-rust-preferred-not-absolute.md` and
//! `docs/adr/0007-testing-conventions.md`), not as a library other crates
//! depend on, so most of its surface has no real external consumer.
//!
//! The library target exists so `benches/` (a separate compiled crate,
//! unlike inline `#[cfg(test)]` modules) can link against the handful of
//! modules actually benchmarked; only those are `pub`, along with the
//! modules their public types unavoidably reference in field/method
//! signatures. Keeping the rest `pub(crate)` avoids `clippy::pedantic`'s
//! public-API-surface lints (`must_use_candidate`, `missing_errors_doc`,
//! `missing_panics_doc`, `missing_debug_implementations`) firing crate-wide
//! for code that was never meant to be a public API.

/// Rolling pre-buffer of recent frames/audio (see `RingBuffer`).
///
/// Public: used by `benches/` to exercise ring-buffer throughput, and its
/// types (`TimestampedFrame`, `TimestampedAudio`) appear in `recorder`'s
/// public API.
pub mod buffer;
/// Camera and audio capture callbacks.
pub(crate) mod capture;
/// Pure timing/bookkeeping state for a single recorded clip. Public: appears
/// in `recorder::RecordingEvent`'s public API.
pub mod clip_state;
/// CLI argument parsing.
pub(crate) mod config;
/// Repeat-sighting confirmation gate for YOLO detections.
pub(crate) mod confirmation;
/// YOLO object-detection inference. Public: used by `benches/` to exercise
/// preprocess/postprocess.
pub mod detect;
/// Recording-event state machine and shutdown/close/seed-drain lifecycle.
pub(crate) mod event_lifecycle;
/// ffmpeg subprocess helpers (video encoder, audio mux, resampling). Public:
/// used by `benches/` to exercise frame resampling, and appears in
/// `recorder`'s implementation surface.
pub mod ffmpeg;
/// Camera liveness/stall detection (no camera dependency).
pub(crate) mod liveness;
/// Logging setup (`init_logging`, `TeeWriter`).
pub(crate) mod logging;
/// Background-subtraction motion gate. Public: used by `benches/` to
/// exercise the motion gate.
pub mod motion;
/// Shared `OpenCV` conversion helpers.
pub(crate) mod opencv_utils;
/// Output file/folder naming. Public: appears in `recorder`'s public API.
pub mod paths;
/// Opt-in live preview window.
pub(crate) mod preview;
/// Camera reconnect gating and stream-rebuild mechanism.
pub(crate) mod reconnect;
/// Recording lifecycle and ffmpeg-backed encoding. Public: used by
/// `benches/` to exercise frame resampling via `RecordingEvent`.
pub mod recorder;
/// Clip `.json` sidecar output shapes (ADR 4). Public: appears in
/// `recorder`'s public API.
pub mod sidecar;
/// Startup dependency checks.
pub(crate) mod startup;
/// Test fixtures shared across more than one module's test suite.
#[cfg(test)]
pub(crate) mod test_support;
/// Per-poll start/extend trigger decisions driving the recording lifecycle.
pub(crate) mod triggering;
/// YOLO-detection-to-trigger evaluation.
pub(crate) mod triggers;
/// Shutdown handshake primitive between the detection and writer threads.
pub(crate) mod writer_drained;

/// Top-level wiring: `run()` and the three long-lived worker loops.
///
/// Public: `src/main.rs` is a separate crate from this library even within
/// the same package, so it needs `pub` visibility to call `app::run()`.
pub mod app;

/// Milliseconds in one second, used to convert a frames-per-second rate into
/// a poll interval (`1000 / fps`).
pub(crate) const MILLIS_PER_SEC: u64 = 1000;
