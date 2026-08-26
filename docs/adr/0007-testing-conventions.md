# 7. Testing conventions: `cargo-llvm-cov`, inline `#[cfg(test)]`, and the test-module lint allowlist

## Status

Accepted. The "binary-only, no `src/lib.rs`" premise below no longer holds as of
ADR 8: a library target was added so `benches/` can link against the crate. This does
not change the decision recorded here (tests still live inline as
`#[cfg(test)] mod tests`, not in a top-level `tests/` directory); see "Amendment" at
the end of this ADR for why the reasoning still holds even though its "no `src/lib.rs`"
premise is now out of date.

## Context

The project had no test suite (`#[test]` was unused in the entire crate) before the
work this ADR documents. Several decisions had to be made once, rather than
re-litigated per file/PR: which coverage tool to use, where tests should physically
live given the crate's structure, and how test code should relate to the project's
otherwise very strict clippy lint policy (`Cargo.toml` denies `clippy::all`,
`pedantic`, `nursery`, `cargo`, plus explicit denies on `unwrap_used`,
`indexing_slicing`, `arithmetic_side_effects`, and more, with zero escape hatches in
production code).

## Decision

### Coverage tool: `cargo-llvm-cov`, not `cargo-tarpaulin`

`cargo-llvm-cov` uses LLVM's native source-based coverage instrumentation (the same
mechanism `rustc`/`cargo` ship) rather than tarpaulin's ptrace-based approach.
tarpaulin's ptrace mode has known issues with subprocess-spawning tests and FFI
boundaries, both of which this crate has throughout (`recorder.rs` spawns real
`ffmpeg` subprocesses — `detect.rs`/`motion.rs`/`opencv_utils.rs` cross into ONNX
Runtime and OpenCV via FFI). `cargo-llvm-cov` is actively maintained, integrates with
`cargo test` directly, and produces `--fail-under-lines`-style gates usable in CI
alongside human-readable HTML/lcov output for local debugging. See ADR 6 for how its
stable-toolchain exclusion granularity (file-level only) shaped where untestable code
physically lives.

### Test placement: inline `#[cfg(test)] mod tests`, no `tests/` directory

Every test (unit, OpenCV-backed, ffmpeg-integration, or `#[ignore]`'d
model-dependent) lives in an inline `#[cfg(test)] mod tests` block at the bottom of
the file it exercises, compiled as part of the crate's own test harness. There is
deliberately no top-level `tests/` directory.

At the time this decision was made, the crate was binary-only (`src/main.rs`, no
`src/lib.rs`), and Rust's `tests/*.rs` integration tests compile as separate crates
that can only import symbols from a crate's *library* target, so a binary-only crate
exposed nothing for `tests/` to link against. Adding a `lib.rs` purely to unlock a
`tests/` directory was considered and rejected at the time: every test target
identified during planning was reachable as an inline `#[cfg(test)]` module against
the binary's own compiled test harness, so the added indirection would have bought
nothing.

**A `src/lib.rs` was later added for an unrelated reason (ADR 8: unlocking real
`cargo bench` support).** This ADR's placement decision was revisited at that point
and reaffirmed, not reversed: see "Amendment" below for why inline `#[cfg(test)]`
modules remained the right choice even once a `tests/`-capable library target existed.
`RecordingEvent::start`, `Detector::load`, and every other ffmpeg-/OpenCV-/ONNX-Runtime-
backed constructor continue to be exercised through the same compiled artifact the
shipped binary uses (via the library crate `src/main.rs` now consumes), not a separate
build path a `tests/*.rs` file would introduce.

### Clippy-in-tests allowlist

Every `#[cfg(test)] mod tests` block starts with a module-level doc comment (one
sentence, satisfies `missing_docs`) followed by a scoped `#![allow(...)]` covering
only the lints that module's *own* test code actually needs, with a `reason = "..."`
explaining why:

```rust
#[cfg(test)]
mod tests {
    //! Tests for ... (one line)
    #![allow(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        clippy::missing_panics_doc,
        reason = "test assertions favor unwrap/indexing for clarity — panics here fail \
                   the test, which is the intended behavior"
    )]

    use super::*;
    // ...
}
```

`unwrap_used`, `indexing_slicing`, and `missing_panics_doc` are needed in every test
module (test assertions routinely `.unwrap()` a `Result`/`Option` to fail loudly on
the actual bug rather than laundering it through `?`, and index fixed-size test data
directly). Two more lints were needed only in specific files, added there rather than
to every module unconditionally:

- `clippy::arithmetic_side_effects` and `clippy::cast_possible_wrap`: needed by
  `detect.rs`'s synthetic-YOLO-output-tensor test helper, which does plain arithmetic
  (`84 * num_anchors`) and `usize as i64` casts against small hardcoded test
  dimensions, not the proof-obligation style (`#[allow(..., reason = "bound proven
  above")]` comments) production code uses for the same operations.
  `clippy::cast_possible_wrap` is `pedantic`-level, everything else here is
  `clippy::all`/`nursery`-level.
- `clippy::unchecked_time_subtraction`: needed by `main.rs`'s stall/expiry tests,
  which backdate `Instant`s directly (`Instant::now() - Duration::from_secs(30)`)
  rather than sleeping in real time to test threshold logic — test durations are small
  hardcoded constants, so underflow past the Unix epoch is not reachable.

Each file's allowlist is scoped to only what that file's tests actually trigger,
confirmed by running `cargo clippy --all-targets -- -D warnings` after writing each
module rather than copy-pasting a maximal list everywhere pre-emptively — a narrower,
per-file list makes it visible in review when a *new* test in that file starts doing
something structurally different from what came before.

Integration-style tests that spawn real `ffmpeg` or load the real ONNX model are
**not** exempted from any additional lints beyond the above — `RecordingEvent`,
`Detector`, `MotionGate`, etc. are exercised through their real, undecorated public
API exactly as production code calls them, so no additional test-only relaxation was
needed to write them.

### `#[ignore]` for model-dependent tests, not a `local-only` feature flag

Tests that need `models/yolov8n.onnx` (gitignored, not committed; see the project
guidance file's "Runtime dependencies" section) or a working ONNX Runtime build use `#[ignore =
"<reason and how to run explicitly>"]` rather than a cargo feature flag or a build
script check. `cargo test` skips them by default (matching what CI can run); `cargo
test -- --ignored` (or `--include-ignored` for the full suite) runs them locally on a
machine that has both dependencies. This was chosen over a feature flag because
`#[ignore]` requires no `Cargo.toml` changes, no conditional compilation, and no
separate CI matrix entry to reason about: the tests always compile, they simply don't
run without an explicit opt-in flag, which keeps `cargo test`/`cargo clippy
--all-targets` behaving identically whether or not the model file happens to be
present.

## Consequences

- A new test file follows an established, three-part pattern: inline `#[cfg(test)]`
  module, a scoped `#![allow(...)]` header sized to that file's actual needs, and
  (only if it touches `Detector`/`MotionGate` together, or `Detector` alone)
  acquisition of `detect::MODEL_TEST_LOCK` before doing anything else — see ADR 6 for
  why that lock exists.
- Running the full test suite locally with the model present
  (`cargo test -- --include-ignored`) takes several seconds longer than the CI-only
  subset, dominated by ONNX Runtime session construction — this is expected and not a
  target for optimization, since these tests run rarely (locally, before a release)
  rather than on every CI push.
- If this crate ever gains a genuine library consumer (unlikely given ADR 1's
  single-binary distribution model), a `src/lib.rs` split would become worth
  revisiting; at that point a `tests/` directory would become usable and this
  decision should be re-examined against the tradeoffs recorded here.

## Amendment: `src/lib.rs` exists now, but the placement decision still stands

ADR 8 added `src/lib.rs` so `benches/` (a separate compiled crate, exactly like
`tests/*.rs`) could link against the modules actually benchmarked. This technically
removes the original blocker cited above: `tests/*.rs` integration tests could now
compile and link, the same way `benches/*.rs` does.

The placement decision itself was re-examined at that point and reaffirmed:

- Every test that exists today was written against the inline `#[cfg(test)]`
  convention, and every one of them still compiles and runs exactly as before; moving
  them to `tests/*.rs` would only relocate passing tests, not enable any new one.
- Inline tests sit next to the code they exercise, which this project has relied on
  throughout (see the clippy-in-tests allowlist section above, written and reviewed
  file-by-file against code sitting directly above each test module). A `tests/`
  directory would separate that pairing for no offsetting benefit.
- Most modules stayed `pub(crate)` in `lib.rs` specifically to keep clippy's
  public-API-surface lints from firing crate-wide (see ADR 8). A `pub(crate)` module's
  private items are invisible to a `tests/*.rs` file (a separate crate) but fully
  visible to an inline `#[cfg(test)] mod tests` (compiled as a submodule in the same
  crate). Moving to `tests/*.rs` now would force many more modules to `pub`, reopening
  the exact clippy fallout ADR 8 scoped down.

Net effect: `src/lib.rs`'s existence is orthogonal to this ADR's decision. Tests
continue to live inline; `benches/` is the only thing under this crate that compiles
as a genuinely separate crate against the library target.
