# 2. Single trigger path: motion gate + YOLO confirmation

## Status

Accepted (supersedes an earlier version of this ADR that proposed a separate
door-zone trigger path; see "Revision" below). The single-trigger-path decision below
still holds, but "correctness comes entirely from the YOLO confirmation requirement" no
longer describes actual behavior as of the repeat-sighting confirmation gate added
after this ADR was written — see "Amendment" below.

## Context

The system needs to record real motion events — including someone opening or closing
a door — while explicitly not false-triggering on indoor sources of repetitive motion
such as ceiling fans or flickering light.

An earlier version of this decision introduced a second, door-specific trigger path
(a manually-configured "door zone" rectangle where motion alone, with no YOLO
confirmation, would trigger recording). On review, that was unnecessary complexity:
a door opening is not a distinct technical case requiring its own code path — it's
simply real, non-repetitive motion, which the motion gate is already meant to catch
and distinguish from repetitive motion like a fan. Adding a separate "door zone"
concept solved a problem that didn't exist and required the user to manually measure
and enter pixel coordinates for no real benefit over the existing gate.

## Decision

There is one trigger path:

A background-subtraction motion gate (OpenCV MOG2/KNN) runs continuously across the
full frame. When it trips, YOLO inference runs to confirm whether a living subject
(person, or any COCO animal class) is actually present. **Recording only starts on a
confirmed YOLO classification** — never on the gate alone.

This means:

- A ceiling fan, curtain flutter, or lighting flicker can trip the gate all day
  without ever producing a recording, because YOLO has no trained concept of a fan
  resembling a living thing.
- A door opening or closing is exactly the kind of large, real, non-repetitive motion
  the gate is designed to catch; no door-specific handling is needed for the gate
  itself to register it as motion worth investigating. (Whether a door-open-with-
  nobody-visible event should itself produce a recording without any living subject
  present was not requested and remains out of scope — the current design only
  records when YOLO confirms a subject.)

The gate's only purpose is avoiding wasted inference cycles on an empty room;
correctness against false positives comes entirely from the YOLO confirmation
requirement. (See "Amendment" below: in practice, a *single* confirmed YOLO
classification turned out not to be trustworthy enough on its own, and a repeat-sighting
requirement was added on top of this without changing the single-trigger-path structure
described here.)

Detection scope is not limited to person/cat/dog: it allowlists person plus every
animal class already present in COCO (bird, cat, dog, horse, sheep, cow, elephant,
bear, zebra, giraffe), since the goal is "every living thing," and COCO already
covers this with no additional model training or engineering cost.

## Revision

The original version of this ADR proposed a second "door-zone path": user-configured
rectangular regions where motion alone (no YOLO confirmation) would trigger recording,
intended to handle door-open/close events specifically. This was removed because it
never actually detected a door — it only skipped YOLO confirmation within a
manually-drawn rectangle, which added a config option, a struct, and validation code
without solving a problem the motion gate didn't already solve. The stated goal
("catch a door opening/closing, don't false-trigger on a fan") is fully met by the
single motion-gate-plus-YOLO-confirmation design above.

## Consequences

- The motion module (`motion.rs`) only needs to report whole-frame motion; there is
  no separate zone-based reporting or configuration.
- The JSON sidecar for each recorded clip records the confirmed classes and their
  timestamps/confidence; there is no "trigger path" distinction to record since there
  is only one path.
- No per-deployment zone configuration (pixel coordinates, room-specific setup) is
  required from the user.

## Amendment: single-poll YOLO confirmation is not sufficient on its own

In production use, `yolov8n` was observed to hallucinate a living-thing class on a
single frame of a static, empty scene at confidence scores spanning the same range as
genuine detections (up to 0.83 confidence on an empty room) — meaning no
`--detection-confidence` threshold can separate a real detection from noise by score
alone, contradicting this ADR's original claim that YOLO confirmation alone is
sufficient for correctness.

This did **not** reopen the single-trigger-path question this ADR settles — the motion
gate plus YOLO confirmation remains the only trigger path, and no door-zone-style bypass
was reintroduced. Instead, a repeat-sighting requirement was layered on top of the existing
YOLO confirmation step: the same living-thing class must be seen on a second poll within
a short window (`PENDING_CONFIRMATION_WINDOW`, currently 5s) before it counts, both to
start a new recording and to extend an already-active one's post-buffer window. A
one-off hallucinated frame doesn't repeat; a real subject, which keeps tripping the
motion gate and getting re-detected for as long as it's in frame, does. See
`confirm_pending`/`PendingConfirmation` in `main.rs` for the implementation.
