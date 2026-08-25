# 1. Rust preferred, not absolute

## Status

Accepted

## Context

motioncap is a webcam-based security tool: continuous capture, motion/object
detection, and video+audio recording. It's meant to run as a long-lived background
process and to be distributed as a binary to other people running unknown hardware.
Rust was the starting preference, but two capabilities needed during design have no
mature pure-Rust implementation at the required quality bar:

- Real-time, hardware-accelerated video encoding (H.264, using NVENC or a platform's
  hardware encoder).
- Background-subtraction motion modeling (MOG2/KNN-style).

## Decision

Rust is the default for all application logic: camera/audio capture, buffering,
detection glue, path/file handling, and the recorder/event lifecycle. Two exceptions
use an established C/C++ library instead of a pure-Rust implementation:

- **Video encoding** uses the system `ffmpeg` binary (invoked as a subprocess).
  Pure-Rust mp4 *muxing* exists, but there is no mature pure-Rust *encoder* with
  hardware acceleration — writing one would mean re-implementing NVENC/VAAPI/Pi
  hardware-encoder integration from scratch. This is close to a hard technical wall,
  not a preference.
- **Background subtraction** uses OpenCV's `BackgroundSubtractorMOG2`/`KNN` via the
  `opencv` crate (bindings, not a rewrite). A pure-Rust hand-rolled running-average
  background model was a real, considered alternative here — this exception was
  chosen deliberately, prioritizing battle-tested accuracy/reliability over avoiding
  one extra system dependency, not because Rust is incapable of the math.

Both are declared as hard runtime requirements, checked at startup (see ADR 5).

## Consequences

- Users must have `ffmpeg` and OpenCV installed on their system; neither is bundled
  into the binary.
- The `opencv` crate builds bindings against whatever OpenCV version is present via
  pkg-config at compile time (verified working against OpenCV 5.0 on the development
  machine, via `opencv5.pc` — note this is a newer major version than the `opencv4`
  pkg-config name many guides assume).
- If a pure-Rust background-subtraction implementation matures later, revisiting the
  OpenCV dependency is reasonable; the ffmpeg dependency is expected to remain for the
  foreseeable future given the encoding-acceleration gap.
- ADR 8 added a `src/lib.rs` library target alongside `src/main.rs`, so `cargo bench`
  (via Criterion) can link against the crate. This does not change anything decided
  here: the shipped artifact is still exactly one binary users run, built the same way
  (`cargo build --release`); `src/lib.rs` only changes how that binary's own source is
  organized internally, and most of it stays `pub(crate)` rather than becoming a
  real external API (see ADR 8).
