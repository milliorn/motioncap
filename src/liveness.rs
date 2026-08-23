use std::time::Duration;

use crate::clip_state;

/// Tracks the most recent frame timestamp `run_detection_loop` has evaluated,
/// and how long that timestamp has been unchanged, to detect a stalled camera
/// before any recording has started (see `frame_liveness_advanced`).
pub struct FrameLiveness {
    /// The last frame timestamp actually evaluated.
    pub(crate) timestamp: std::time::Instant,
    /// When `timestamp` was first observed to still be the latest frame.
    pub(crate) unchanged_since: std::time::Instant,
    /// Whether the stall warning has already been logged for this
    /// `timestamp`, so a still-stalled camera logs once per episode instead
    /// of once per poll.
    pub(crate) warned: bool,
}

impl FrameLiveness {
    /// How long `timestamp` has been the latest frame seen, as of `now`.
    /// Shared by every threshold checked against this stall (see
    /// `frame_liveness_advanced`, `reconnect::maybe_reconnect_camera`) so they
    /// all read one computation over `unchanged_since` instead of each
    /// re-deriving it.
    pub(crate) fn stalled_for(&self, now: std::time::Instant) -> Duration {
        now.duration_since(self.unchanged_since)
    }
}

/// Updates `last_seen` for a newly-polled `latest_frame` timestamp and
/// reports whether the loop should proceed with it. Returns `false` once the
/// same timestamp has recurred for `clip_state::MAX_FRAME_STALL`, i.e.
/// `latest_frame()` is returning a frame the camera delivered a while ago,
/// not a fresh one, which otherwise would get silently re-run through
/// motion/YOLO on every poll and cascade into duplicate recordings. A
/// same-timestamp recurrence shorter than that is ordinary jitter between
/// polls and camera delivery, not a stall, so it's allowed through unlogged.
/// The stall is logged once (not once per poll) when it's first detected.
pub fn frame_liveness_advanced(
    last_seen: &mut Option<FrameLiveness>,
    frame_timestamp: std::time::Instant,
) -> bool {
    let now = std::time::Instant::now();

    match last_seen {
        Some(seen) if seen.timestamp == frame_timestamp => {
            if seen.stalled_for(now) < clip_state::MAX_FRAME_STALL {
                return true;
            }

            if !seen.warned {
                log::warn!(
                    "camera appears stalled: no new frame since {:?}; skipping detection ticks until it recovers",
                    seen.timestamp
                );
                seen.warned = true;
            }

            false
        }
        _ => {
            *last_seen = Some(FrameLiveness {
                timestamp: frame_timestamp,
                unchanged_since: now,
                warned: false,
            });

            true
        }
    }
}

/// Called after `reconnect::maybe_reconnect_camera` rebuilds the stream, in
/// place of clearing `last_seen` to `None`.
///
/// The ring buffer only evicts frames on `push_frame`, so `latest_frame()`
/// can keep returning the pre-reconnect stale frame until the rebuilt stream
/// delivers its first one. Clearing to `None` would let
/// `frame_liveness_advanced` treat that same stale frame as freshly "live" on
/// the very next tick (its `_` match arm resets unconditionally), skipping
/// the `MAX_FRAME_STALL` grace period entirely and re-running motion/YOLO on
/// stale data. Keeping `timestamp` as-is means a still-stale frame is still
/// recognized as unchanged, while resetting `unchanged_since` to now restarts
/// both the `MAX_FRAME_STALL` pause clock and (via `stalled_for`) the
/// `CAMERA_RECONNECT_STALL` clock, so a rebuilt stream that stalls again
/// immediately doesn't instantly qualify for another reconnect attempt.
/// `warned` is also reset so the next stall episode logs again.
pub fn reset_liveness_after_reconnect(last_seen: &mut Option<FrameLiveness>) {
    if let Some(seen) = last_seen {
        seen.unchanged_since = std::time::Instant::now();
        seen.warned = false;
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the pure decision logic backing camera stall detection.
    #![allow(
        clippy::unwrap_used,
        clippy::missing_panics_doc,
        clippy::unchecked_time_subtraction,
        reason = "test assertions favor unwrap/plain time arithmetic for clarity; test durations \
                   are small hardcoded constants, so underflow is not reachable"
    )]

    use std::time::{Duration, Instant};

    use super::*;

    // --- frame_liveness_advanced ---

    #[test]
    fn frame_liveness_advanced_true_on_first_observation() {
        let mut last_seen = None;
        let ts = Instant::now();

        assert!(frame_liveness_advanced(&mut last_seen, ts));
        assert!(last_seen.is_some());
    }

    #[test]
    fn frame_liveness_advanced_true_on_new_timestamp() {
        let mut last_seen = Some(FrameLiveness {
            timestamp: Instant::now(),
            unchanged_since: Instant::now(),
            warned: true,
        });

        let new_ts = Instant::now() + Duration::from_millis(100);
        assert!(frame_liveness_advanced(&mut last_seen, new_ts));
        // Resets tracking for the new timestamp.
        assert_eq!(last_seen.unwrap().timestamp, new_ts);
    }

    #[test]
    fn frame_liveness_advanced_true_on_same_timestamp_within_stall_threshold() {
        let ts = Instant::now();
        let mut last_seen = Some(FrameLiveness {
            timestamp: ts,
            unchanged_since: Instant::now(),
            warned: false,
        });

        assert!(frame_liveness_advanced(&mut last_seen, ts));
    }

    #[test]
    fn frame_liveness_advanced_false_and_warns_once_past_stall_threshold() {
        // clip_state::MAX_FRAME_STALL is a hardcoded const (1.5s), not
        // injectable, so this test genuinely waits it out rather than
        // asserting the boundary logic through a shortened duration.
        let ts = Instant::now();
        let mut last_seen = Some(FrameLiveness {
            timestamp: ts,
            unchanged_since: Instant::now()
                - clip_state::MAX_FRAME_STALL
                - Duration::from_millis(50),
            warned: false,
        });

        assert!(!frame_liveness_advanced(&mut last_seen, ts));
        assert!(last_seen.as_ref().unwrap().warned);

        // A second poll past the threshold must not re-log (warned stays true,
        // not reset), matching "logged once per episode, not once per poll".
        assert!(!frame_liveness_advanced(&mut last_seen, ts));
        assert!(last_seen.unwrap().warned);
    }

    // --- reset_liveness_after_reconnect ---

    #[test]
    fn reset_liveness_after_reconnect_resets_warned_and_clock_but_not_timestamp() {
        let original_ts = Instant::now() - Duration::from_secs(30);
        let mut last_seen = Some(FrameLiveness {
            timestamp: original_ts,
            unchanged_since: original_ts,
            warned: true,
        });

        reset_liveness_after_reconnect(&mut last_seen);

        let seen = last_seen.unwrap();
        assert_eq!(seen.timestamp, original_ts);
        assert!(!seen.warned);
        assert!(seen.unchanged_since > original_ts);
    }

    #[test]
    fn reset_liveness_after_reconnect_is_noop_on_none() {
        let mut last_seen: Option<FrameLiveness> = None;
        reset_liveness_after_reconnect(&mut last_seen);
        assert!(last_seen.is_none());
    }
}
