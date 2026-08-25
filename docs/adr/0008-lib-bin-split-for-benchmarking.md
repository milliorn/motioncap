# 8. Library target added alongside the binary, scoped to keep clippy's public-API bar narrow

## Status

Accepted

## Context

`cargo bench` was requested for this project. The standard, idiomatic way to write
real benchmarks on stable Rust is Criterion, and Criterion benchmarks live in a
`benches/` directory, compiled as separate crates from the main package (the same
mechanism as `tests/*.rs`; see ADR 7). A separate crate can only link against another
crate's *library* target. Before this decision, motioncap had no `src/lib.rs`
(binary-only, per the structure ADR 7 was written against), so `benches/*.rs` had
nothing to link against, exactly the same wall ADR 7 hit for `tests/*.rs`.

Two questions had to be answered together, not separately: how to make `cargo bench`
possible at all, and whether doing so was actually a legitimate architectural need or
scope creep unrelated to what benchmarking requires. A programmer reasonably asked: if
unlocking a benchmarking tool requires a nontrivial refactor, is that a sign the
project's original binary-only structure was wrong, and is there a cleaner way to
structure a "standalone binary" project in Rust generally?

The answer settled on: a `lib.rs` + thin `main.rs` split is the standard shape for any
nontrivial Rust binary (ripgrep, fd, bat, and most CLI tools of comparable complexity
all use it), and it does not change what ships. `cargo` supports a package containing
both a library target and a binary target simultaneously; the binary target becomes a
thin entry point that calls into the library. The end product is still exactly one
binary a user runs (`cargo build --release`, then run the resulting executable) -
nothing about distribution, installation, or how users invoke motioncap changes.
Binary-only was never required to produce a standalone executable; it was simply the
structure this project happened to start with.

Attempting the naive version of this split (`mod X;` -> `pub mod X;` for every
existing module, verbatim) surfaced a second, larger problem: `clippy::pedantic`
(denied crate-wide per `Cargo.toml`) enforces a materially stricter bar on items that
are part of a crate's actual public API (`must_use_candidate`,
`missing_errors_doc`, `missing_panics_doc`, `missing_debug_implementations`) than on
items only reachable within a single binary crate. Making every module `pub`
surfaced roughly 130 new clippy errors across the codebase, on functions and types
that were `pub` in name only (never reachable from outside the compiled binary) before
this change, and were never intended as a real external API surface.

## Decision

**Add `src/lib.rs`, but keep almost every module `pub(crate)`; make a module `pub`
only when a bench file actually needs to reach it, or another `pub` module's public
signatures unavoidably reference it.**

Concretely:

- `src/main.rs` shrank to a six-line entry point:
  `fn main() -> anyhow::Result<()> { motioncap::app::run() }`.
- A new `src/app.rs` holds `run()` and the three long-lived worker loops
  (`run_detection_loop`, `run_recording_writer_loop`, `run_preview_loop`), moved
  verbatim out of the old `main.rs`. It stays `pub` in `lib.rs`'s module declaration
  only because `src/main.rs` is a separate compiled crate from the library even within
  the same package, and needs `pub` visibility to call `app::run()`.
- `src/lib.rs` declares every other module `pub(crate)` by default. Only `buffer`,
  `clip_state`, `detect`, `ffmpeg`, `motion`, `paths`, `recorder`, and `sidecar` are
  `pub`: `motion`, `detect`, `buffer`, and `recorder` because `benches/` exercises them
  directly; `clip_state`, `ffmpeg`, `paths`, and `sidecar` because `recorder`'s public
  types (`RecordingEvent`, `RecordingEventParams`) reference their types in field or
  method signatures, and a `pub` item cannot expose a `pub(crate)`-only type without
  triggering a reachability error.
- Every clippy finding the newly-`pub` modules exposed was fixed for real (not
  suppressed): `#[must_use]` on pure accessors/constructors, `# Errors` sections on
  every `pub fn` returning `Result`, `# Panics` sections on every `pub fn` that can
  panic, and `Debug` implementations on every newly-`pub` type. Three types
  (`MotionGate`, `Detector`, `RecordingEvent`) wrap a field that itself doesn't
  implement `Debug` (`opencv::core::Ptr`, `ort::session::Session`,
  `std::process::Child` respectively) and got a manual `impl Debug` reporting their
  other fields via `finish_non_exhaustive()`, rather than a blanket
  `#[allow(missing_debug_implementations)]`.
- `benches/buffer.rs` was added as the first real benchmark, exercising
  `RingBuffer::push_frame`/`latest_frame`/`frames_since` (the hot paths every
  detection/writer/preview poll hits) via Criterion, with a scoped
  `#![allow(missing_docs, clippy::missing_docs_in_private_items, ...)]` at the top of
  the file: `criterion_group!`/`criterion_main!` generate an undocumented `fn main`
  that the file itself has no way to document, since it comes from macro expansion.

This was chosen over two alternatives considered and rejected:

- **Making every module fully `pub` and fixing/allowing all ~130 findings crate-wide.**
  Rejected because it would have meant either doing ~130 call sites of doc/attribute
  work unrelated to what benchmarking actually needed (most of those modules are never
  touched by a bench file), or blanket-allowing `clippy::pedantic`'s public-API lints
  at the crate root, which would have weakened the zero-escape-hatch lint policy
  `Cargo.toml` documents for the whole codebase, for the sake of code that was never
  meant to be a public API in the first place.
- **`benches/*.rs` `#[path]`-including source files directly, avoiding a `lib.rs`
  entirely.** Considered but rejected: this duplicates compilation of any included
  file across every bench binary that includes it (no shared compilation unit the way
  a library target provides), and risks silent drift if an included file ever grows a
  dependency on something only `main.rs`/`app.rs` provides. A real library target,
  scoped down with `pub(crate)`, was judged the more maintainable and more idiomatic
  choice once the clippy-fallout problem had a real fix.

## Consequences

- `cargo bench` now works exactly as documented upstream: Criterion's HTML reports,
  statistical warm-up/sampling, and run-over-run regression comparison all function
  against the real `RingBuffer` (and, as more benchmarks are added, `MotionGate`,
  `Detector`, `RecordingEvent`) from the library crate.
- Adding a new benchmark that needs a currently-`pub(crate)` module requires promoting
  that module (and any modules its public types reference) to `pub` in `lib.rs`, then
  fixing whatever `clippy::pedantic` findings that promotion surfaces on that module's
  own items, the same way this decision did for the first four. This is expected,
  ongoing cost of adding bench coverage, not a one-time migration.
- `cargo-llvm-cov`'s coverage report and thresholds are unaffected in substance by this
  split: every source line still belongs to exactly one file, `cargo llvm-cov` still
  instruments the library and binary targets together, and the aggregate percentage
  moved only in the sense that `main.rs`'s prior line count (mostly the loop functions,
  now in `app.rs`) moved with it. See ADR 6's amendment for the specific per-file
  breakdown that changed.
- ADR 7's decision to keep tests inline (`#[cfg(test)] mod tests`, no `tests/`
  directory) was re-examined once `src/lib.rs` existed and reaffirmed, not reversed;
  see ADR 7's amendment. `benches/` remains the only thing in this crate compiled as a
  genuinely separate crate against the library target.
- If a future benchmark needs a module currently kept `pub(crate)` for reasons beyond
  "nothing needs it public yet" (none currently exist), that would be a signal to
  revisit this ADR rather than reflexively promoting it to `pub`.
