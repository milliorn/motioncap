# 6. Coverage exclusions: file-level via `cargo-llvm-cov`, plus a dedicated `coverage_excluded.rs`

## Status

Accepted

## Context

The project added a test suite with `cargo-llvm-cov` as its coverage tool, targeting the
highest coverage genuinely achievable, not a symbolic percentage, but a number where
every uncovered line traces to a documented, concrete reason. Two separate obstacles
stood between the codebase and 100%:

1. **Irreducible untestable code.** Some functions can only run against real hardware,
   an unrepeatable process-global side effect, or an external model file, and no
   automated test can fake that safely:
   - `capture::camera::start_camera_capture` and the auto-detect branch of
     `resolve_camera_index`: open a real `/dev/videoN` device via `nokhwa`.
   - `capture::audio::start_audio_capture`: opens the real default audio input device
     via `cpal` (`default_host`/`default_input_device`/`default_input_config`/
     `build_input_stream`/`stream.play`). The per-sample format conversion
     (`samples_to_f32`) and the format-support check (`sample_format_supported`) that
     this function's `match` relies on are pulled out into their own pure functions and
     unit-tested independently, the same split as `resolve_camera_index` below.
   - `preview.rs` (every method): drives `OpenCV`'s highgui, which needs a real
     X11/Wayland display — there is no headless fake for a GUI window.
   - `coverage_excluded::init_logging`: calls `env_logger::Builder::init()`, which sets
     the process-global logger and panics if called a second time in the same
     process. `cargo test` runs the entire suite in one process, so this can only be
     invoked once, in the real startup path, without an order-dependent risk of
     poisoning every other test's use of the `log::` macros.
   - `coverage_excluded::run` (the former body of `main`) and `coverage_excluded::run_detection_loop`,
     top-level wiring that opens the real camera/audio devices and constructs a real
     `Detector::load` (model file + ONNX Runtime session) unconditionally before doing
     anything else, so even their shutdown-only paths can't be exercised without that
     hardware/model dependency.
   - `coverage_excluded::run_recording_writer_loop` and `coverage_excluded::run_preview_loop`,
     testable up to their shutdown-check branch (a plain `AtomicBool`, no hardware
     needed; both have a unit test exercising exactly that), but their steady-state
     loop body waits on real wall-clock time via `thread::sleep` and then either drains
     real ring-buffer/ffmpeg state or drives `PreviewWindow`, which needs a real
     `OpenCV` highgui display.
   - `coverage_excluded::maybe_reconnect_camera`, which calls
     `capture::camera::start_camera_capture` once past its threshold/cooldown guard,
     opening a real `/dev/videoN` device. The guard itself (`should_reconnect`) is pure
     and stays in `main.rs`, unit-tested directly there.
   - `Config::parse_args`: reads the real process's `std::env::args()`.
   - `check_ffmpeg`'s "not found" branch (`startup/depcheck.rs`), only reachable if
     `ffmpeg` is genuinely absent from `PATH` — this crate denies `unsafe_code`, and
     `std::env::set_var` (the only way to fake `PATH` from within a test) requires
     `unsafe` on current Rust, so this branch is not safely fakeable from a test.
   - A handful of `?`-propagated error branches (`Detector::load`/`detect`'s `ort_err`
     calls, `RecordingEvent::finish`'s ffmpeg-exit-status and rename-failure paths) that
     would require deliberately corrupting a model file, killing a subprocess mid-run,
     or forcing a filesystem rename to fail: fault injection disproportionate to the
     value of covering an already-simple `bail!`/`.context()` call.

2. **A tooling limitation that makes the *reported* number worse than the *real*
   coverage, if not designed around.** `cargo-llvm-cov` can mark code excluded two
   ways: the `#[coverage(off)]` attribute, which excludes individual lines/functions
   in place, or `--ignore-filename-regex`, which excludes entire files. The former
   requires `#![feature(coverage_attribute)]` (**confirmed directly against this
   crate's toolchain (`rustc 1.97.1`, stable) that it fails to compile** with
   `error[E0658]: the #[coverage] attribute is an experimental feature`. Since this
   project uses only stable Rust everywhere else (no nightly in `Cargo.toml`, CI, or
   a `rust-toolchain` file — see ADR 1's preference for boring, portable tooling), only
   `--ignore-filename-regex`'s whole-file granularity was available.

The naive approach (leaving every untestable function in `main.rs` alongside the
dozens of thoroughly-tested ones, then either accepting `main.rs`'s reported number
sitting in the 50s, or excluding all of `main.rs` by file) was rejected. The first
option is discouraging and looks indistinguishable from "nobody tried." The second
throws away real signal: `main.rs` was roughly half genuinely-tested pure decision
logic (`confirm_pending`, `expire_stale_pending`, `frame_liveness_advanced`,
`should_reconnect`, `ActiveEvent`, `WriterDrained`, `close_event_if_done`,
`finish_event_on_shutdown`, `evaluate_active_event`, `try_start_recording`, ...) — a
whole-file exclusion would make a regression in any of that logic invisible to the
coverage gate.

## Decision

**Isolate every untestable function into its own module, `src/coverage_excluded.rs`,** so
that file-level exclusion becomes precise instead of a blunt trade-off. `coverage_excluded::run`
(top-level wiring, formerly `main`'s body), `coverage_excluded::init_logging`,
`coverage_excluded::run_detection_loop`, `coverage_excluded::run_recording_writer_loop`,
`coverage_excluded::run_preview_loop`, and `coverage_excluded::maybe_reconnect_camera` live
there; `main.rs`'s own `fn main()` shrinks to a one-line call into `coverage_excluded::run()`.
The latter three moved out of `main.rs` after initially landing there: each is
untestable past its own guard/shutdown-check branch for the same reasons as the rest of
this module (real wall-clock polling, a real `OpenCV` window, or a real camera device),
and leaving them in `main.rs` meant those specific lines dragged its reported number
down for no different a reason than the functions already isolated here. Every pure
decision function the detection loop (or these newly-moved functions) calls
(`try_start_recording`, `evaluate_active_event`, `close_event_if_done`,
`finish_event_on_shutdown`, `frame_liveness_advanced`, `should_reconnect`,
`seed_and_drain_active_event`, `confirm_pending`, `expire_stale_pending`, ...) stays in
`main.rs`, made `pub(crate)` where `coverage_excluded.rs` needs to call it, and is
unit-tested directly there.

Not every function with an untestable branch moved, though: `try_start_recording` and
`evaluate_active_event` still call `RecordingEvent::start` (spawns ffmpeg) on their
confirmed-detection path, which no test reaches, since doing so honestly (not just to
satisfy the coverage tool) requires a real photo of a living-thing subject for YOLO to
confirm, the same category of dependency as `detect.rs`'s `#[ignore]`'d tests. That gap
was deliberately left in place rather than extracted: splitting the ffmpeg-spawning tail
out of either function would fragment cohesive decision logic (which classes to record,
whether to start at all) away from its own tests purely to move a handful of lines,
exactly the "erasing real coverage signal" trade-off this ADR's Context section already
rejected once. A `?`-propagated error branch or a genuinely hardware-bound function
(opens a device, drives a GUI, blocks on real time) is worth isolating; a function that's
merely *sometimes* reached only under real inputs a test can't fabricate is not.

`cargo llvm-cov`'s CI/local invocation excludes exactly three files by regex:

```sh
--ignore-filename-regex 'coverage_excluded\.rs|preview\.rs|capture/camera\.rs'
```

`capture/camera.rs` and `capture/audio.rs` are *not* excluded wholesale; only the
specific hardware-bound functions within them are marked with a `// coverage:
excluded: <reason>` source comment. `resolve_camera_index`'s `Some(path)` branch (pure
string parsing, no hardware) is unit-tested normally and pulls `capture/camera.rs`'s
reported percentage up from what a full-file exclusion would otherwise hide;
`capture/camera.rs` still appears in the regex above because the *rest* of that file
(the auto-detect branch and `start_camera_capture` itself) is hardware-bound, so
excluding the whole file remains the least-bad option there. `capture/audio.rs`, by
contrast, has no function whose reported percentage is worth keeping under file-level
exclusion once `samples_to_f32`/`sample_format_supported` were pulled out and tested;
`start_audio_capture` itself stays uncovered (marked with the same source comment) but
the file as a whole is left in the coverage report rather than excluded, currently
landing around 40% since that function is most of the file's line count. The comment
convention exists because the tool itself can't express "exclude this function, not
this file" on stable; the comment is what makes the omission visible in source and
in code review, even though the coverage tool can't enforce it at that granularity.

Every `#[ignore]`'d test (there are 9, split across `detect.rs` and `main.rs`) that
constructs a real `Detector`/ONNX Runtime session and/or a real `MotionGate` shares one
`detect::MODEL_TEST_LOCK` mutex, acquired for the test's full duration before doing
anything else. This was added after directly reproducing a heap-corruption abort
("corrupted double-linked list") when `cargo test`'s default parallelism ran multiple
such tests concurrently — neither `ort`'s `Session` nor `OpenCV`'s
`BackgroundSubtractorMOG2` are documented as safe to construct/run concurrently across
independent instances in separate threads, and this crate's own production code never
does so (YOLO inference and the motion gate both run single-threaded inside the
detection worker, by design). The lock is `#[cfg(test)]`-only and adds no runtime cost
to the shipped binary.

Two numeric thresholds are enforced, not one, because CI and local runs have
genuinely different achievable ceilings:

- **Local** (`cargo llvm-cov --workspace --ignore-filename-regex '...' -- --include-ignored`,
  run on a machine with `models/yolov8n.onnx` and a working ONNX Runtime build already
  present): **93.46%** measured after `capture/audio.rs` moved from full exclusion to
  the partial-exclusion convention (down from 95.72%, not because anything regressed,
  but because `start_audio_capture`'s uncovered lines are now counted against the total
  instead of hidden by file-level exclusion; the file's own reported percentage rose,
  from 0% to ~40%, which is the real signal this move was for). The remaining gap is
  the `?`-propagated error branches described above, `start_audio_capture` itself, plus
  `try_start_recording`/`evaluate_active_event`'s confirmed-detection path, which needs
  a real photo of a living-thing subject to reach honestly (see the Decision section).
- **CI** (`cargo llvm-cov --workspace --fail-under-lines 76 --ignore-filename-regex '...'`,
  no `--ignored`, the gitignored model file is never fetched in CI): **76.38%**
  measured, gated at 76 (lowered from 77 for the same reason as the local number above)
  with a small margin for measurement noise. This is meaningfully lower than the local
  number specifically because the 9 `#[ignore]`'d tests (which exercise a large fraction
  of `detect.rs`'s and `main.rs`'s logic) never run there; that gap is expected and
  documented, not a sign CI is under-testing relative to what it can actually run.

## Consequences

- `main.rs` and every file except the three fully-excluded ones are held to a *real*,
  gate-enforced 100%-or-explained-gap standard; a regression in any tested function
  moves the reported number and fails CI, not just `coverage_excluded.rs`'s untestable
  wiring. `capture/audio.rs` is included in that same real standard for the part of it
  that's actually testable (`samples_to_f32`, `sample_format_supported`), even though
  its overall file percentage stays well under 100% because of `start_audio_capture`.
- Adding a new function to `coverage_excluded.rs` is a deliberate signal: it should only
  ever contain thin sequencing/wiring calling into functions defined (and tested)
  elsewhere. If a function there starts accumulating real decision logic, that logic
  belongs in `main.rs`, not `coverage_excluded.rs` — see that module's own doc comment.
- The CI threshold (76) and local threshold (93.46%, informal, not currently gated by
  a CI job, since there is no CI runner with the model file present) must be
  re-verified and adjusted together whenever the untestable-function list changes;
  they are not meant to converge, since the gap between them is structural (CI never
  has the model file), not a discrepancy to eliminate.
- If `#[coverage(off)]` stabilizes on a future stable Rust release, `coverage_excluded.rs`
  could in principle be dissolved back into `main.rs` with the exclusion moved to
  precise `#[coverage(off)]` attributes per-function, but this is not planned — the
  module split has its own value as a visible boundary between "wiring" and "logic"
  independent of the coverage tooling that motivated it.
