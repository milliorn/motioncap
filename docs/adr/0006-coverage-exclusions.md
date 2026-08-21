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
`config_coverage_excluded.rs`, and `startup/depcheck_coverage_excluded.rs` were all
deleted; their contents moved back into `main.rs`, `motion.rs`, `recorder.rs`,
`capture/camera.rs`, `capture/audio.rs`, `config.rs`, and `startup/depcheck.rs`
respectively. Two reasons drove the reversal, not one:

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

The sections below describe the sibling-file pattern in place from its introduction
until this ADR's reversal above. The reasoning about *why* `cargo-llvm-cov` only offers
whole-file exclusion on stable Rust (the two-obstacle Context section above) is still
accurate; only the response to that constraint changed.

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

`cargo llvm-cov`'s CI/local invocation excluded files by regex:

```sh
--ignore-filename-regex 'coverage_excluded\.rs|preview\.rs'
```

`coverage_excluded\.rs` (unanchored) matched both the crate-root `coverage_excluded.rs`
and any `*_coverage_excluded.rs` sibling file, so `capture/camera_coverage_excluded.rs`,
`capture/audio_coverage_excluded.rs`, `recorder_coverage_excluded.rs`, and
`motion_coverage_excluded.rs` were covered by this single pattern without needing their
own entries.

`capture/camera.rs`, `capture/audio.rs`, `config.rs`, `recorder.rs`, and
`startup/depcheck.rs` all used the same convention: rather than leaving their
hardware-/process-bound functions in place under a source comment, each moved them
entirely into a sibling file (`capture/camera_coverage_excluded.rs`,
`capture/audio_coverage_excluded.rs`, `config_coverage_excluded.rs`,
`recorder_coverage_excluded.rs`, `startup/depcheck_coverage_excluded.rs`), mirroring
the crate-root `coverage_excluded.rs` split. `recorder_coverage_excluded.rs` differed
from the other siblings in one respect: it held two `impl RecordingEvent` methods
(`start`, `finish`) rather than free functions, since Rust allows a type's `impl` block
to be split across files in the same crate, and `RecordingEvent`'s fields needed to
become `pub` (module-private field access doesn't cross file boundaries the way it
crosses `mod` boundaries within one file) for the sibling file's `impl` block to
construct/consume them. `startup/depcheck_coverage_excluded.rs` differed in the opposite
direction from the others: the function that moved (`check_dependencies`) was not itself
hardware-bound, only its call site into an untestable sibling function
(`check_ffmpeg`, which stayed behind in `depcheck.rs` since its own body is fully
testable) made it unreachable in the `Err` case.

`motion_coverage_excluded.rs` differed from every other sibling above: it was the whole
of what was `motion.rs` (`MotionGate::new`/`evaluate`, `changed_ratio`, and their
tests), renamed wholesale rather than split, since `MotionGate`/`changed_ratio`'s
`?`-propagated error arms are checked at the call site of a `Result`-returning function
regardless of which file the callee itself lives in; moving only the two-or-three-line
irreducible OpenCV calls into a wrapper function in a sibling file, while leaving the
calling functions (and their `?` call sites) in `motion.rs`, was tried and measured not
to remove the uncovered *Regions* from `motion.rs`'s count (see the "current" Decision
section above for the correction: this was true for Regions, but Lines, the metric
actually gated, did reach 100% via that same wrapping, a distinction not recognized at
the time this section was originally written).

Two numeric thresholds were enforced, not one, because CI and local runs had genuinely
different achievable ceilings:

- **Local**: last measured at 97.51% total *line* coverage (`main.rs` itself at 94.44%,
  the lowest of any non-excluded file; every other non-excluded file at 100%).
- **CI**: **81.86%** line coverage measured, gated at 81 with a small margin for
  measurement noise. (Previously gated at 77/77.28%, then briefly at 83 after a
  recalibration that read the Regions column, 84.38%, instead of the Lines column the
  gate actually checks; that mistake made the 83 gate fail CI outright since real Lines
  coverage was only 81.86%, and was caught and fixed the same day the 83 value was set.)

## Consequences (superseded)

- `main.rs` and every file except the fully-excluded `coverage_excluded.rs`/
  `preview.rs`/`capture/camera_coverage_excluded.rs`/`capture/audio_coverage_excluded.rs`/
  `config_coverage_excluded.rs`/`recorder_coverage_excluded.rs`/
  `motion_coverage_excluded.rs`/`startup/depcheck_coverage_excluded.rs` were held to a
  *real*, gate-enforced 100%-or-explained-gap standard; a regression in any tested
  function moved the reported number and failed CI, not just those excluded files'
  untestable wiring.
- Adding a new function to `coverage_excluded.rs` was a deliberate signal: it should only
  ever contain thin sequencing/wiring calling into functions defined (and tested)
  elsewhere.
- If `#[coverage(off)]` stabilizes on a future stable Rust release, `coverage_excluded.rs`
  could in principle be dissolved back into `main.rs` with the exclusion moved to
  precise `#[coverage(off)]` attributes per-function; this reasoning still applies to
  the current, merged-file state as well (see the "current" Consequences section above).
