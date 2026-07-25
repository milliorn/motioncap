# 4. Storage layout and retention policy

## Status

Accepted

## Context

Recorded clips need to be organized in a way that's browsable without a big flat dump
of files, and searchable later without re-running inference on archived footage.
Retention (auto-deleting old clips) was also considered.

## Decision

**Layout:** date-then-event. Recordings are written to
`<output_dir>/<YYYY-MM-DD>/<YYYY-MM-DD_HH-MM-SS>_<class1>_<class2>.mp4`, i.e. one folder
per calendar day containing files named by the full local date and time the event
started plus every distinct class detected during the clip, alphabetized (e.g.
`2026-07-24_14-32-05_cat_person.mp4` for a clip containing both a cat and a person at
different points). The date is repeated in the filename, not just the containing
folder, so that filenames sort chronologically by name alone in a flat listing across
multiple days, not only within a single day's folder. A clip that starts before
midnight stays whole in the start-day's folder rather than being split at the day
boundary.

**Metadata sidecar:** each clip gets a same-named `.json` file recording per-event
detail: which trigger path fired (living-thing vs. door-zone, see ADR 2), timestamps
of each detection within the clip (offset from clip start), detected classes, and
confidence scores. This exists specifically to support future search/filtering (e.g.
"show me every clip with a dog") and debugging false positives/negatives without
needing to re-run inference on old video — the sidecar preserves the expensive-to-
recompute information at negligible cost to write.

**Retention:** explicitly out of scope for v1. The user manages disk space manually.
Automatic pruning was considered and deferred rather than rejected outright — it's a
reasonable v2 addition once actual storage/usage patterns from real deployment are
known, rather than guessing at a retention policy upfront.

## Consequences

- No disk-space safety net exists in v1; a long-running deployment could fill its
  disk if unattended. This is an accepted, explicit tradeoff, not an oversight.
- The date-then-event layout makes bulk day-based archival/deletion easy to do
  manually (e.g. `rm -rf <output_dir>/2026-01-*`) even without built-in retention.
- Filename-based multi-class listing means a future search UI could work purely by
  parsing filenames for simple cases, but the sidecar remains the source of truth for
  anything needing timestamps or confidence.
- Repeating the date in the filename makes each filename a few characters longer, but
  keeps clips independently sortable/identifiable outside the context of their folder
  (e.g. after being copied elsewhere, or viewed in a flat listing).
