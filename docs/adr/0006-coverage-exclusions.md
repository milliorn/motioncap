# 6. Coverage exclusions: file-level via `cargo-llvm-cov`, kept to a single `preview.rs`

## Status

Accepted. Supersedes an earlier version of this same decision (below, kept for its
still-accurate reasoning about *why* `cargo-llvm-cov`'s tooling constraints exist) that
isolated every untestable function into a dedicated `*_coverage_excluded.rs` sibling
file. That sibling-file pattern was reversed: it grew to 8 extra files (one per
untestable function/small group of functions), which felt like sprawl disproportionate
to the problem, and it hid the codebase's real CI-reachable coverage ceiling behind a
number (81.86%) that looked meaningfully better than the honest one. All of that code
moved back into its natural home file (`main.rs`, `motion.rs`, `recorder.rs`,
`capture/camera.rs`, `capture/audio.rs`, `config.rs`, `startup/depcheck.rs`,
`detect.rs`); `preview.rs` is the only file still excluded, since GUI code genuinely has
no automatable path to coverage on this tool. See the Decision/Consequences sections
below for the current state and thresholds; the Context section immediately following
still accurately explains why `--ignore-filename-regex` is the only exclusion mechanism
available on stable Rust, which remains true and relevant.

## Context

**Note:** this section is retained from the superseded decision and describes the
codebase as it stood under the `*_coverage_excluded.rs` sibling-file pattern; module
paths like `motion_coverage_excluded::` and `coverage_excluded::` below reflect that
historical layout, not where this code lives today (see "Decision (current)" for the
present module paths and coverage numbers). It's kept because the *reasoning* about why
`--ignore-filename-regex` is the only exclusion mechanism available on stable Rust is
still accurate and relevant.

The project added a test suite with `cargo-llvm-cov` as its coverage tool, targeting the
highest coverage genuinely achievable, not a symbolic percentage, but a number where
every uncovered line traces to a documented, concrete reason. Two separate obstacles
stood between the codebase and 100%:

1. **Irreducible untestable code.** Some functions can only run against real hardware,
   an unrepeatable process-global side effect, or an external model file, and no
   automated test can fake that safely:
   - `capture::camera_coverage_excluded::start_camera_capture` and
     `auto_detect_camera_index`: open a real `/dev/videoN` device via `nokhwa`.
   - `capture::audio_coverage_excluded::start_audio_capture`: opens the real default
     audio input device via `cpal` (`default_host`/`default_input_device`/
     `default_input_config`/`build_input_stream`/`stream.play`). The per-sample format
     conversion (`samples_to_f32`) and the format-support check
     (`sample_format_supported`) that this function's `match` relies on live in
     `capture::audio` instead, as their own pure functions, unit-tested independently.
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
     `capture::camera_coverage_excluded::start_camera_capture` once past its threshold/cooldown guard,
     opening a real `/dev/videoN` device. The guard itself (`should_reconnect`) is pure
     and stays in `main.rs`, unit-tested directly there.
   - `config_coverage_excluded::parse_args`: reads the real process's `std::env::args()`.
   - `check_ffmpeg`'s "not found" branch, only reachable if `ffmpeg` is genuinely absent
     from `PATH`, since this crate denies `unsafe_code`, and `std::env::set_var` (the
     only way to fake `PATH` from within a test) requires `unsafe` on current Rust, so
     this branch is not safely fakeable from a test. The underlying probe
     (`check_ffmpeg_probe`, parameterized on the binary name) is itself fully tested
     directly, including its not-found branch reached by probing a name that genuinely
     isn't on `PATH`. `check_ffmpeg`'s own hardcoded call into it with the literal
     `"ffmpeg"` stays in `startup/depcheck.rs` alongside `check_ffmpeg_probe` and
     `check_model_file`, since that call itself always succeeds in a test (real `PATH`
     has `ffmpeg`) and so isn't the untestable line - only the caller that would
     observe a failure is. `check_dependencies` (`startup/depcheck.rs` originally) is
     that caller: its `check_ffmpeg()?` has an error-propagation branch reachable only
     if `check_ffmpeg` itself returns `Err`, which no test can produce, so it moved
     wholesale into `startup/depcheck_coverage_excluded.rs`, taking its two existing
     tests with it, mirroring the `RecordingEvent::start`/`finish` split described
     below: `cargo-llvm-cov` attributes an uncovered `?` region to its own source line
     regardless of which function the fallible call lives in, so only moving the entire
     function containing the branch removes it from `depcheck.rs`'s count.
   - A handful of `?`-propagated error branches (`Detector::load`/`detect`'s `ort_err`
     calls) that would require deliberately corrupting a model file: fault injection
     disproportionate to the value of covering an already-simple `bail!`/`.context()`
     call.
   - `motion_coverage_excluded::MotionGate::new`/`evaluate` and the `changed_ratio`
     helper they call: three of their four `?`-propagated error arms come from real
     `OpenCV` calls confirmed empirically (not assumed) to have no
     externally-inducible failure mode with any input this crate can construct
     (`create_background_subtractor_mog2` never errored across every parameter
     tried, and `BackgroundSubtractorMOG2::apply` tolerated an empty, mismatched-shape,
     and mismatched-type `Mat`). The fourth (`changed_ratio`'s own `count_non_zero`
     call) genuinely is reachable and is directly tested; it's just never reached via
     `evaluate`, since `apply`'s output mask is always single-channel regardless of
     input. Unlike the other entries in this file, this one moved wholesale (whole
     functions and their tests, not just the irreducible lines) as an interim measure,
     since wrapping only the fallible calls was tried and measured not to remove the
     regions from `motion.rs`'s count (see below).
   - `recorder_coverage_excluded::RecordingEvent::start`/`finish` and the two free
     functions they call (`spawn_video_encoder`, `mux_audio_into_video`): each contains
     at least one `Command::spawn`/`.output()` call failing to exec `ffmpeg` at all (as
     opposed to running and exiting nonzero, a separate and fully-tested arm), or
     `finish`'s `Child::wait` returning `Err`, which std documents as only reachable if
     the process was already reaped elsewhere, a condition this module's exclusive
     ownership of its own `Child` rules out. `recorder.rs` originally carried this gap
     in place (`RecordingEvent::finish`'s ffmpeg-exit-status, rename-failure, and
     sidecar-write paths, plus `write_frame`/`write_audio`'s broken-pipe/closed-handle
     paths); nearly all of it turned out to be reachable with fault injection that's
     cheap and safe in this specific case (killing the child `ffmpeg` process directly,
     swapping in a read-only file handle, pre-creating a directory at a rename/write
     target) and is covered by tests that stayed in `recorder.rs`. Only the exec-failure
     and `wait`-`Err` arms above remained irreducible; since llvm-cov attributes an
     uncovered region to the `?`/branch's own source line regardless of which function
     the fallible call lives in, moving just those two or three calls into a wrapper
     (leaving `start`/`finish` themselves in `recorder.rs`) does not remove the
     region from `recorder.rs`'s count, only moving the *entire* function containing
     the branch does. So `start`, `finish`, `spawn_video_encoder`, and
     `mux_audio_into_video` moved wholesale into `recorder_coverage_excluded.rs`
     (as a second `impl RecordingEvent` block plus two free functions), taking their
     existing passing tests with them; every other `RecordingEvent` method (`seed`,
     `drain_frames`, `drain_audio`, `write_frame`, `write_audio`, `record_detection`,
     `touch`, `record_motion`, `quiet_for`, `camera_stalled`) and all of `ClipState`
     stayed in `recorder.rs`, which now reports genuine 100% coverage (lines,
     functions, and regions) rather than 100%-minus-a-documented-gap.

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

## Decision (current)

**Every function lives in its natural home file; only `preview.rs` is excluded by
file.** The `*_coverage_excluded.rs` sibling-file pattern described in the superseded
Decision below (further down this document) was reversed: `coverage_excluded.rs`,
`motion_coverage_excluded.rs`, `recorder_coverage_excluded.rs`,
`capture/camera_coverage_excluded.rs`, `capture/audio_coverage_excluded.rs`,
`config_coverage_excluded.rs`, `startup/depcheck_coverage_excluded.rs`, and
`detect_coverage_excluded.rs` were all deleted; their contents moved back into
`main.rs`, `motion.rs`, `recorder.rs`, `capture/camera.rs`, `capture/audio.rs`,
`config.rs`, `startup/depcheck.rs`, and `detect.rs` respectively. Two reasons drove the
reversal, not one:

1. **File-count sprawl.** 8 separate files existed purely to carve out a handful of
   untestable functions each; `config_coverage_excluded.rs` held exactly one function
   (`parse_args`) in an 18-line file. The file-level-only exclusion mechanism this ADR's
   Context section documents (still true, see below) means this sprawl has no fix that
   preserves fine-grained exclusion on stable Rust; the only real lever was to stop
   paying for the precision where the codebase didn't need it as badly as it needed
   fewer files.
2. **A hidden ceiling.** `motion_coverage_excluded.rs` in particular had drifted from
   this ADR's own stated model (isolate only the irreducible lines) into a full-file
   exclusion hiding ~175 lines of pure, already-tested `MotionGate`/`changed_ratio`
   logic, invisible to the coverage *report*, though never gated behind `#[ignore]`
   (all 4 of its tests ran in every `cargo test`). More broadly, the CI-gated number
   (81.86%, see the superseded Decision's threshold section) looked meaningfully
   healthier than what CI could actually exercise, because the genuinely hardware-bound
   code (camera/audio device access, the detection/recording/preview loops) was excluded
   from the count entirely rather than showing up as a real, honest gap the way
   `main.rs`'s untestable lines already did before any of this pattern existed.

Investigated and rejected alternatives to plain file merging:

- **`#[coverage(off)]`**: still nightly-only (`#![feature(coverage_attribute)]`),
  reconfirmed against this crate's toolchain; unchanged from the superseded Decision's
  finding.
- **`grcov`'s regex-based `--excl-line`/`--excl-start`/`--excl-stop` markers**: real,
  stable-compatible per-line exclusion exists in this tool, but switching coverage
  tools for a comment-marker convenience was judged not worth it given `tarpaulin`
  below turned up a concrete reason to be cautious about ptrace-based tools generally
  in this codebase.
- **`tarpaulin`'s `#[cfg(not(tarpaulin_include))]`**: a genuine stable cfg attribute,
  evaluated specifically for this project and rejected: its ptrace-based instrumentation
  has a documented history (upstream issue #704) of segfaults and `SIGCHLD` deadlocks in
  concurrent native-library scenarios, which is exactly the failure class
  `detect::MODEL_TEST_LOCK` (see below) already exists to prevent. It also needs a
  `--no-dead-code` flag just to build against the `opencv` crate's FFI bindings, which
  its own maintainer states makes results less accurate, undercutting the reason to
  switch tools in the first place.

One important correction to the superseded Decision's own reasoning, discovered while
re-verifying it during this reversal: its claim that wrapping only the two irreducible
OpenCV calls in `MotionGate` "was tried and measured not to remove the regions from
`motion.rs`'s count" is true only for the **Regions** column. The `?`-propagation branch
marker is attributed to the call site regardless of where the callee lives, so wrapping
never moves Regions coverage, but the **Lines** column (the only one
`--fail-under-lines` gates) does reach 100% via that wrapping, since the call-site line
itself still counts as executed. Confirmed directly: `motion.rs`, `recorder.rs`, and
`startup/depcheck.rs` all report genuine 100% Lines today with their irreducible `Err`
arms in place, unmoved, because the call-site line runs regardless of which branch it
takes.

`preview.rs` remains excluded by file, unchanged from the superseded Decision: it drives
a real `OpenCV` highgui window, which has no automatable path to coverage on any tool
available to this project.

Every `#[ignore]`'d test (10 total, across `detect.rs` and `main.rs`) that constructs a
real `Detector`/ONNX Runtime session and/or a real `MotionGate` still shares one
`detect::MODEL_TEST_LOCK` mutex, unchanged from the superseded Decision; this
requirement is orthogonal to the file-layout reversal and remains fully in effect.

Two numeric thresholds are enforced, not one, because CI and local runs have genuinely
different achievable ceilings:

- **Local** (`cargo llvm-cov --workspace --ignore-filename-regex 'preview\.rs' --
  --include-ignored`, run on a machine with `models/yolov8n.onnx` and a working ONNX
  Runtime build already present): last measured at **87.89%** total *line* coverage.
  `motion.rs`, `recorder.rs`, and `startup/depcheck.rs` are at genuine 100%;
  `capture/audio.rs` (40.00%) and `capture/camera.rs` (50.67%) are the lowest, since
  their hardware-opening functions (`start_audio_capture`, `start_camera_capture`,
  `auto_detect_camera_index`) have no automatable test path even locally; `main.rs`
  (79.95%) and `detect.rs` (96.33% locally, with the model file present) carry the rest
  of the gap. Re-run the command above to get the exact current number rather than
  trusting a stale figure here.
- **CI** (`cargo llvm-cov --workspace --fail-under-lines 73 --ignore-filename-regex
  'preview\.rs'`, no `--include-ignored`, the gitignored model file is never fetched in
  CI): **73.52%** line coverage measured, gated at 73 with a small margin for
  measurement noise. This is meaningfully lower than the local number both because the
  10 `#[ignore]`'d tests never run there (as before) and because the file-merge above
  newly exposes `capture/audio.rs`/`capture/camera.rs`/`main.rs`'s hardware-bound gaps to
  the CI number for the first time; that combined gap is expected and honest, not a sign
  of a real regression in what's tested. (Previously gated at 81, informally reaching as
  high as 83/84 at points in this ADR's history; see the superseded Decision below for
  that number's own trajectory. The drop to 73 reflects newly-counted code, not newly
  broken tests.)

## Amendment: `main.rs` split into `lib.rs` + `app.rs` (ADR 8)

ADR 8 moved `run`/`run_detection_loop`/`run_recording_writer_loop`/`run_preview_loop`
(previously described throughout this ADR as living in `main.rs`) into a new
`src/app.rs`, part of a new library target; `src/main.rs` shrank to a three-line
`fn main() { motioncap::app::run() }`. This is a pure file move, not a coverage-policy
change: every reference above to "`main.rs`'s loop functions" or "`main.rs`'s
untestable lines" now means the equivalent functions in `app.rs`. The reasoning for
why those functions can't be automated (real camera/audio devices, real wall-clock
`thread::sleep`, a real `OpenCV` highgui display) is unchanged; only the file name
changed.

Concretely, after the move:

- `main.rs` itself is now three lines, trivially 0% or 100% covered depending on
  whether the one delegation line executes; it carries no meaningful signal of its own
  and was never the interesting number this ADR tracks.
- `app.rs` inherits the coverage profile this ADR previously attributed to `main.rs`:
  its two loop functions with a testable shutdown-check branch (unit-tested, matching
  the tests described above) and untestable steady-state bodies waiting on real
  hardware/wall-clock time.
- The two `#[cfg(test)]` tests previously described as living in `main.rs`'s test
  module (`run_recording_writer_loop_drains_and_signals_on_immediate_shutdown`,
  `run_preview_loop_returns_immediately_on_shutdown_without_preview`) moved with their
  functions and now live in `app.rs`'s test module.
- Aggregate coverage numbers (local and CI) are materially unchanged by this move
  alone: the same lines are covered or uncovered as before, just filed under a
  different filename in the per-file report. Re-run the coverage commands in this ADR
  to get current numbers rather than trusting the figures recorded above, which
  predate this split.

## Consequences (current)

- Only `preview.rs` is excluded from the coverage report by file. Every other file,
  including the ones with the largest real gaps (`capture/audio.rs`,
  `capture/camera.rs`, `main.rs`), reports its true number; a regression anywhere moves
  that file's percentage and, in aggregate, the CI gate.
- The CI threshold (73) and last recorded local line coverage (87.89%, informal, not
  currently gated by a CI job, since there is no CI runner with the model file present)
  must be re-verified and adjusted together whenever significant code is added to the
  hardware-bound loops/capture functions; they are not meant to converge, since the gap
  between them is structural (CI never has the model file, camera, audio device, or
  display), not a discrepancy to eliminate.
- Raising the CI number now requires writing real tests against previously-hidden gaps
  (particularly `capture/camera.rs`/`capture/audio.rs`'s hardware-opening functions and
  `main.rs`'s loop functions), not moving code into an excluded file. The largest
  concrete opportunities, in order of missed-line count: `main.rs` (618 missed),
  `detect.rs` (73 missed, mostly closed already by the `#[ignore]`'d tests once a model
  file is available), `capture/audio.rs` (54 missed), `capture/camera.rs` (37 missed).
- If `#[coverage(off)]` stabilizes on a future stable Rust release, this file-merge
  decision could be revisited in favor of precise per-line exclusion in place, but this
  is not currently planned.

---

## Decision (superseded, kept for historical context)

The prior decision isolated every untestable function into a dedicated
`*_coverage_excluded.rs` sibling file per source file (`coverage_excluded.rs`,
`motion_coverage_excluded.rs`, `recorder_coverage_excluded.rs`,
`capture/camera_coverage_excluded.rs`, `capture/audio_coverage_excluded.rs`,
`config_coverage_excluded.rs`, `startup/depcheck_coverage_excluded.rs`,
`detect_coverage_excluded.rs`), excluded from
coverage via a single regex (`--ignore-filename-regex
'coverage_excluded\.rs|preview\.rs'`, unanchored so it matched every sibling). This made
file-level exclusion precise instead of a blunt whole-file trade-off, at the cost of the
8-file sprawl and hidden ceiling described in the "Decision (current)" section above;
see that section's two reasons for the reversal.

Two details from that era are worth preserving since they're easy to get wrong if
re-attempted:

- **`RecordingEvent::start`/`finish` required `pub` fields.** Splitting an `impl` block
  across files works in Rust, but module-private field access doesn't cross file
  boundaries the way it crosses `mod` boundaries within one file, so
  `recorder_coverage_excluded.rs`'s `impl RecordingEvent` block needed the struct's
  fields to be `pub`, not just `pub(crate)`. Reversing the split let those fields go back
  to private (see the current `src/recorder.rs`).
- **Wrapping only the fallible call, not the whole function, doesn't help the Regions
  column, but it does help Lines**, the column `--fail-under-lines` actually gates. This
  ADR's original text claimed wrapping just the irreducible OpenCV calls in `motion.rs`
  "was tried and measured not to remove the regions," which is true only for Regions;
  Lines reaches 100% either way, since the call-site line itself counts as executed
  regardless of which branch it takes. `motion.rs` was moved wholesale anyway, before
  this distinction was recognized.

Numeric thresholds at the time: local coverage was last measured at 97.51%; CI was gated
at **81** (81.86% measured, up from an earlier 77/77.28%, with one same-day-caught
mistake where a recalibration briefly set the gate to 83 by misreading the Regions
column instead of Lines).

## Consequences (superseded)

Every file except the excluded siblings and `preview.rs` was held to a genuine
100%-or-explained-gap standard, so a regression anywhere else still failed CI.
`coverage_excluded.rs` itself was meant to hold only thin sequencing/wiring, never real
logic; a deliberate signal for what belonged there versus in `main.rs`.
