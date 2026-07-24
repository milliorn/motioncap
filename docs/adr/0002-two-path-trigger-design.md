# 2. Two independent trigger paths for recording

## Status

Accepted

## Context

The system needs to record when (a) a person or animal is present, and (b) when a
door/structural event happens (e.g. a door opening), while explicitly not
false-triggering on indoor sources of repetitive motion such as ceiling fans or
flickering light. A single object detector (YOLO) can recognize living things but has
no concept of "a door," since that's not an object class — it's a state change.

## Decision

Recording is triggered by two independent, differently-gated paths:

1. **Living-thing path.** A background-subtraction motion gate (OpenCV MOG2/KNN) runs
   continuously across the full frame. When it trips, YOLO inference runs to confirm
   whether a living subject (person, or any COCO animal class) is actually present.
   **Recording only starts on a confirmed YOLO classification** — never on the gate
   alone. This means a ceiling fan, curtain flutter, or lighting flicker can trip the
   gate all day without ever producing a recording, because YOLO has no trained
   concept of a fan resembling a living thing. The gate's only purpose on this path is
   avoiding wasted inference cycles on an empty room; correctness against false
   positives comes entirely from the YOLO confirmation requirement.

2. **Door-zone path.** Users configure rectangular "door zone" regions in frame
   coordinates. Motion detected *within* a door zone triggers recording directly, with
   no YOLO confirmation required. This is deliberately exempted from the
   confirmation rule because a door swinging open is a large, distinct, non-repetitive
   motion event unlikely to be confused with a fan, and because "door" isn't an object
   class a detector recognizes in the first place — requiring YOLO confirmation here
   would mean door-open events could never trigger a recording at all.

Detection scope for the living-thing path is not limited to person/cat/dog: it
allowlists person plus every animal class already present in COCO (bird, cat, dog,
horse, sheep, cow, elephant, bear, zebra, giraffe), since the goal is "every living
thing," and COCO already covers this with no additional model training or engineering
cost.

## Consequences

- The motion module (`motion.rs`) must report door-zone motion separately from
  whole-frame motion, since they lead to different trigger behavior.
- The JSON sidecar for each recorded clip must record which path triggered it
  (door-zone vs. living-thing-confirmed), and for the living-thing path, which
  classes were confirmed.
- Door zones are a per-deployment configuration setting (room/camera-placement
  specific) and cannot be auto-detected; they must be exposed via config.
- A design risk accepted here: a person entering through a monitored door zone
  produces two overlapping triggers (door-zone motion, then likely a living-thing
  confirmation as they continue into frame). The recorder's event lifecycle (single
  open file per event, extended by any new trigger within the post-buffer window)
  naturally merges these into one clip rather than two.
