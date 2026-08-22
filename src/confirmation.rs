use std::time::Duration;

use anyhow::Result;

use crate::config::Config;
use crate::detect::{self, Detector};
use crate::triggers;

/// How long a first, unconfirmed living-thing detection stays eligible for
/// second-hit confirmation (see `PendingConfirmation`) before it's discarded
/// as noise. `yolov8n` occasionally hallucinates a living-thing class at
/// meaningful confidence on a single static frame of a cluttered/low-light
/// scene with nothing alive in it (observed directly: confidences up to 0.83
/// on an empty room, spanning the same range as genuine detections, so no
/// `detection_confidence` value can separate the two by score alone). A real
/// subject keeps tripping the motion gate and getting re-detected on
/// subsequent polls for as long as it's in frame; a single hallucinated
/// frame does not recur. This window is generous relative to
/// `DETECTION_POLL_INTERVAL` because polls only reach YOLO when the motion
/// gate trips, not every tick, so the second confirming poll may be a second
/// or two behind the first rather than immediately after it.
pub const PENDING_CONFIRMATION_WINDOW: Duration = Duration::from_secs(5);

/// A first, not-yet-confirmed living-thing detection while no recording is
/// active, awaiting a second sighting of the same class within
/// `PENDING_CONFIRMATION_WINDOW` before `run_detection_loop` will actually
/// start a recording (see that constant's docs for why single-poll
/// confirmation isn't trustworthy on its own).
pub struct PendingConfirmation {
    /// The living-thing class seen on the first, unconfirmed poll.
    pub class_name: &'static str,
    /// When `class_name` was last seen: the first, unconfirmed poll, or (once
    /// confirmed) the most recent poll that re-confirmed it. See
    /// `confirm_pending`'s doc comment for why this refreshes on every
    /// confirmed repeat rather than staying fixed at the first sighting.
    pub first_seen: std::time::Instant,
}

/// Clears `pending` once it's gone `PENDING_CONFIRMATION_WINDOW` without a
/// fresh sighting. Called on *every* tripped-motion poll (not only polls
/// where `confirm_pending` itself runs), since `triggers::evaluate` returning
/// `None` (motion tripped but YOLO found no living-thing class) must not
/// leave a stale confirmation sitting unexpired forever: `evaluate_active_event`
/// reads `pending_confirmation.is_some()` on exactly that no-detection path to
/// decide whether bare motion still extends the recording, so if this expiry
/// never got a chance to run, a confirmation from minutes ago would keep
/// reading as "recent" indefinitely, reproducing the sub-threshold-motion
/// clip-never-closes bug this whole gate exists to prevent.
pub fn expire_stale_pending(pending: &mut Option<PendingConfirmation>, now: std::time::Instant) {
    if let Some(p) = pending
        && now.duration_since(p.first_seen) > PENDING_CONFIRMATION_WINDOW
    {
        *pending = None;
    }
}

/// Reduces this poll's confirmed detections against `pending` (the previous
/// unconfirmed or already-confirmed sighting, if any) into either a
/// start/extend-worthy set of detections or an updated `pending` to carry
/// into the next poll.
///
/// Once a class clears the repeat-sighting gate, `pending`'s `first_seen` is
/// refreshed (not cleared) on every subsequent poll that still sees it, so a
/// continuously-present real subject confirms on *every* poll after its
/// first repeat rather than alternating pending/confirmed every other poll.
/// Clearing `pending` back to `None` on each confirmation would force a
/// fresh two-poll cycle every time, silently dropping every other detection
/// for as long as the subject stays in frame (observed directly: a
/// continuously-present person alternated "not yet confirmed" / "confirmed"
/// on literally every poll under the first version of this gate). Only
/// aging out after a full `PENDING_CONFIRMATION_WINDOW` with no sighting at
/// all, not resetting on every confirm, is what actually distinguishes
/// a persisting subject from one-off noise.
///
/// Only class identity is checked for the repeat, not exact detection
/// equality, since YOLO's per-frame confidence for the same real subject
/// naturally varies poll to poll. Requiring an identical score would make
/// genuine repeats fail to match as often as it filters noise.
pub fn confirm_pending(
    pending: &mut Option<PendingConfirmation>,
    detections: Vec<detect::Detection>,
    now: std::time::Instant,
) -> Option<Vec<detect::Detection>> {
    expire_stale_pending(pending, now);

    let repeat_confirmed = pending
        .as_ref()
        .is_some_and(|p| detections.iter().any(|d| d.class_name == p.class_name));

    if repeat_confirmed {
        log::debug!("detection confirmed on repeat sighting");
        // Refresh rather than clear: see doc comment above for why staying
        // "live" (instead of resetting to scratch) is what lets a
        // continuously-present subject confirm on every subsequent poll.
        if let Some(p) = pending {
            p.first_seen = now;
        }

        return Some(detections);
    }

    // No match for the currently-pending class (if any) anywhere in this
    // poll's detections. Only now does it get replaced. Checked against
    // every detection this poll, not just the first, so a still-present
    // class isn't evicted just because a *different* class happens to come
    // first in `detect::postprocess`'s fixed class-allowlist order (both can
    // appear together, e.g. a person and a pet in frame at once).
    if let Some(first) = detections.first() {
        log::debug!(
            "detection '{}' not yet confirmed; awaiting repeat within {PENDING_CONFIRMATION_WINDOW:?}",
            first.class_name
        );

        *pending = Some(PendingConfirmation {
            class_name: first.class_name,
            first_seen: now,
        });
    }

    None
}

/// Runs YOLO against `frame` and reduces the result through `confirm_pending`,
/// on behalf of both `try_start_recording` and `evaluate_active_event`. The
/// two callers differ only in what they do with the resulting
/// `Option<Vec<Detection>>`, not in how a poll gets from a tripped motion
/// gate to a confirmed detection. `expire_stale_pending` runs unconditionally
/// before returning regardless of which branch inside `confirm_pending` was
/// taken (including when `triggers::evaluate` finds nothing and
/// `confirm_pending` is never reached), so callers no longer need to
/// remember that invariant themselves. See `expire_stale_pending`'s doc
/// comment for why skipping it on the no-detection path is the exact bug
/// this gate exists to prevent.
pub fn poll_confirmed_detections(
    detector: &mut Detector,
    config: &Config,
    frame: &image::RgbImage,
    pending: &mut Option<PendingConfirmation>,
) -> Result<Option<Vec<detect::Detection>>> {
    let now = std::time::Instant::now();
    let detections = detector.detect(frame, config.detection_confidence)?;
    let confirmed = triggers::evaluate(detections).and_then(|d| confirm_pending(pending, d, now));

    expire_stale_pending(pending, now);

    Ok(confirmed)
}

#[cfg(test)]
mod tests {
    //! Unit tests for the pure decision logic backing the repeat-sighting
    //! confirmation gate.
    #![allow(
        clippy::unwrap_used,
        clippy::missing_panics_doc,
        clippy::unchecked_time_subtraction,
        reason = "test assertions favor unwrap/plain time arithmetic for clarity; test durations \
                   are small hardcoded constants, so underflow is not reachable"
    )]

    use std::time::Instant;

    use detect::Detection;

    use super::*;

    fn detection(class_name: &'static str) -> Detection {
        Detection {
            class_name,
            confidence: 0.9,
        }
    }

    // --- expire_stale_pending ---

    #[test]
    fn expire_stale_pending_clears_after_window_elapses() {
        let now = Instant::now();
        let mut pending = Some(PendingConfirmation {
            class_name: "person",
            first_seen: now - PENDING_CONFIRMATION_WINDOW - Duration::from_secs(1),
        });

        expire_stale_pending(&mut pending, now);

        assert!(pending.is_none());
    }

    #[test]
    fn expire_stale_pending_keeps_fresh_pending() {
        let now = Instant::now();
        let mut pending = Some(PendingConfirmation {
            class_name: "person",
            first_seen: now,
        });

        expire_stale_pending(&mut pending, now);

        assert!(pending.is_some());
    }

    #[test]
    fn expire_stale_pending_is_noop_on_none() {
        let now = Instant::now();
        let mut pending: Option<PendingConfirmation> = None;

        expire_stale_pending(&mut pending, now);

        assert!(pending.is_none());
    }

    // --- confirm_pending ---

    #[test]
    fn confirm_pending_first_sighting_is_not_confirmed() {
        let mut pending = None;
        let now = Instant::now();

        let result = confirm_pending(&mut pending, vec![detection("person")], now);

        assert!(result.is_none());
        assert!(pending.is_some());
        assert_eq!(pending.unwrap().class_name, "person");
    }

    #[test]
    fn confirm_pending_second_sighting_same_class_within_window_confirms() {
        let mut pending = None;
        let first = Instant::now();
        confirm_pending(&mut pending, vec![detection("person")], first);

        let second = first + Duration::from_secs(1);
        let result = confirm_pending(&mut pending, vec![detection("person")], second);

        assert!(result.is_some());
        // first_seen refreshes on confirm rather than clearing, so a
        // continuously-present subject stays confirmed on every subsequent
        // poll instead of alternating confirmed/unconfirmed.
        assert_eq!(pending.unwrap().first_seen, second);
    }

    #[test]
    fn confirm_pending_second_sighting_different_class_replaces_pending_unconfirmed() {
        let mut pending = None;
        let first = Instant::now();
        confirm_pending(&mut pending, vec![detection("person")], first);

        let second = first + Duration::from_secs(1);
        let result = confirm_pending(&mut pending, vec![detection("dog")], second);

        assert!(result.is_none());
        assert_eq!(pending.unwrap().class_name, "dog");
    }

    #[test]
    fn confirm_pending_sighting_after_window_expires_is_treated_as_fresh() {
        let mut pending = None;
        let first = Instant::now();
        confirm_pending(&mut pending, vec![detection("person")], first);

        let after_expiry = first + PENDING_CONFIRMATION_WINDOW + Duration::from_secs(1);
        let result = confirm_pending(&mut pending, vec![detection("person")], after_expiry);

        assert!(result.is_none());
        assert_eq!(pending.as_ref().unwrap().first_seen, after_expiry);
    }
}
