# 2. Single trigger path: motion gate + YOLO confirmation

## Status

Accepted (supersedes an earlier version of this ADR that proposed a separate
door-zone trigger path; see "Revision" below)

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
requirement.

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
