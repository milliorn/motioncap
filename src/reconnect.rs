use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::buffer::RingBuffer;
use crate::capture;
use crate::clip_state;

/// How long a camera stall (see `FrameLiveness`) must persist before
/// `run_detection_loop` tears down and rebuilds the capture stream, rather
/// than continuing to wait for it to recover on its own.
///
/// Deliberately much longer than `clip_state::MAX_FRAME_STALL` (1.5s): that
/// threshold exists to stop feeding stale frames into detection/recording
/// within a couple of seconds, which is far too trigger-happy to also gate
/// tearing down and reopening the OS camera handle. Doing that on every
/// brief stall would thrash the device and could itself induce more stalls.
/// This threshold instead assumes the camera is genuinely gone (see
/// `capture::camera::start_camera_capture`'s doc comment for why nokhwa
/// never recovers from this on its own) and a full stream rebuild is
/// warranted.
pub const CAMERA_RECONNECT_STALL: Duration = Duration::from_secs(15);

/// Minimum time between reconnect attempts once the camera is believed dead,
/// so a camera that fails to reopen (e.g. genuinely unplugged) doesn't get a
/// reopen attempt on every single detection poll while it's absent.
pub const CAMERA_RECONNECT_COOLDOWN: Duration = Duration::from_secs(10);

/// Tracks the most recent frame timestamp `run_detection_loop` has evaluated,
/// and how long that timestamp has been unchanged, to detect a stalled camera
/// before any recording has started (see `frame_liveness_advanced`).
pub struct FrameLiveness {
    /// The last frame timestamp actually evaluated.
    timestamp: std::time::Instant,
    /// When `timestamp` was first observed to still be the latest frame.
    unchanged_since: std::time::Instant,
    /// Whether the stall warning has already been logged for this
    /// `timestamp`, so a still-stalled camera logs once per episode instead
    /// of once per poll.
    warned: bool,
}

impl FrameLiveness {
    /// How long `timestamp` has been the latest frame seen, as of `now`.
    /// Shared by every threshold checked against this stall (see
    /// `frame_liveness_advanced`, `maybe_reconnect_camera`) so they all read
    /// one computation over `unchanged_since` instead of each re-deriving it.
    fn stalled_for(&self, now: std::time::Instant) -> Duration {
        now.duration_since(self.unchanged_since)
    }
}

/// The shared camera handle `run_detection_loop` polls liveness against and,
/// on a prolonged stall, rebuilds via `maybe_reconnect_camera`. Bundled
/// together (rather than passed as two separate `run_detection_loop`
/// parameters) both because they're always used as a pair at that call site
/// and because `run_detection_loop` is already at clippy's argument-count
/// limit.
pub struct DetectionCamera {
    /// The live capture stream; swapped out in place on reconnect. `None`
    /// only for the brief window between dropping the old stream and
    /// successfully opening the replacement. See `maybe_reconnect_camera`.
    pub handle: Arc<Mutex<Option<nokhwa::threaded::CallbackCamera>>>,
    /// The originally configured device (or `None` for auto-detect), reused
    /// unchanged on every reconnect attempt.
    pub device: Option<std::path::PathBuf>,
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

/// The pure threshold/cooldown decision behind `maybe_reconnect_camera`,
/// extracted so it can be unit-tested without a real camera: `true` only if
/// the current stall has persisted past `CAMERA_RECONNECT_STALL` *and* the
/// last reconnect attempt (if any) was more than `CAMERA_RECONNECT_COOLDOWN`
/// ago, so a camera that stays absent doesn't get a rebuild attempt on every
/// single poll tick while it's gone.
fn should_reconnect(
    last_seen: Option<&FrameLiveness>,
    last_reconnect_attempt: Option<std::time::Instant>,
    now: std::time::Instant,
) -> bool {
    let Some(seen) = last_seen else {
        return false;
    };

    if seen.stalled_for(now) < CAMERA_RECONNECT_STALL {
        return false;
    }

    if let Some(attempted_at) = last_reconnect_attempt
        && now.duration_since(attempted_at) < CAMERA_RECONNECT_COOLDOWN
    {
        return false;
    }

    true
}

/// If the current stall (tracked by `last_seen`) has persisted past
/// `CAMERA_RECONNECT_STALL`, tears down and rebuilds the capture stream (see
/// `capture::camera::start_camera_capture`'s doc comment), respecting
/// `CAMERA_RECONNECT_COOLDOWN` between attempts so a camera that stays absent
/// doesn't get reopened on every single poll tick while it's gone.
/// `last_reconnect_attempt` is updated on every attempt (success or failure),
/// so the cooldown also applies after a success. If the rebuilt stream
/// stalls again immediately, this still waits out the cooldown rather than
/// retrying in a tight loop.
///
/// Returns `true` on a successful rebuild, so the caller can reset
/// `last_seen`: the old stream's last timestamp is meaningless to compare
/// the freshly rebuilt stream's frames against. Opens a real `/dev/videoN`
/// device via `start_camera_capture` once past its threshold/cooldown
/// guard, so it's not exercised by an automated test; the guard itself
/// (`should_reconnect`) is pure and is tested directly below.
pub fn maybe_reconnect_camera(
    last_seen: Option<&FrameLiveness>,
    last_reconnect_attempt: &mut Option<std::time::Instant>,
    camera: &Mutex<Option<nokhwa::threaded::CallbackCamera>>,
    camera_device: Option<&std::path::Path>,
    ring_buffer: &Arc<Mutex<RingBuffer>>,
) -> bool {
    let now = std::time::Instant::now();

    if !should_reconnect(last_seen, *last_reconnect_attempt, now) {
        return false;
    }

    *last_reconnect_attempt = Some(now);

    log::warn!(
        "camera has been stalled for over {CAMERA_RECONNECT_STALL:?}; attempting to rebuild the capture stream"
    );

    // The old stream must be torn down (dropping it sets the wedged capture
    // thread's die flag; see `start_camera_capture`'s doc comment) *before*
    // attempting to open a new one, since both hold the same underlying
    // device node open; otherwise every reopen attempt fails with EBUSY for
    // as long as the old instance is still alive.
    drop(camera.lock().expect("camera lock poisoned").take());

    let rebuilt = capture::camera::start_camera_capture(camera_device, Arc::clone(ring_buffer));

    match rebuilt {
        Ok(new_camera) => {
            *camera.lock().expect("camera lock poisoned") = Some(new_camera);
            log::info!("camera stream rebuilt successfully");
            true
        }
        Err(err) => {
            log::error!("failed to rebuild camera stream: {err:?}");
            false
        }
    }
}

/// `false` while the ring buffer hasn't yet had `pre_buffer_secs` worth of
/// wall-clock time to refill since the capture stream was last rebuilt by
/// `maybe_reconnect_camera`.
///
/// A reconnect drops the old `CallbackCamera` and opens a fresh one (see
/// `maybe_reconnect_camera`'s doc comment), so the ring buffer starts
/// effectively empty at that instant: it only evicts on `push_frame`, but
/// nothing pushes *new* frames into it during the stall, and any frames
/// still sitting in it are the stale pre-stall ones the trigger's pre-buffer
/// snapshot has no use for as genuine lead-in. If a trigger fires before the
/// buffer has had a full `pre_buffer_secs` to refill from the rebuilt
/// stream, `try_start_recording` would seed the new clip with only the
/// handful of seconds actually captured since reconnect, instead of the
/// configured lead-in, producing a clip that opens abruptly mid-action
/// rather than easing in (observed directly: a reconnect completing ~4s
/// before the next confirmed trigger produced a clip with ~4s of pre-roll
/// against a configured 10s). `reconnected_at` is `None` before any
/// reconnect has happened this run, in which case the buffer has had the
/// entire process lifetime to fill and this always returns `true`.
pub fn pre_buffer_ready(
    reconnected_at: Option<std::time::Instant>,
    pre_buffer_secs: u32,
    now: std::time::Instant,
) -> bool {
    let Some(reconnected_at) = reconnected_at else {
        return true;
    };

    now.duration_since(reconnected_at) >= Duration::from_secs(u64::from(pre_buffer_secs))
}

/// Called after `maybe_reconnect_camera` rebuilds the stream, in place of
/// clearing `last_seen` to `None`.
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
    //! Unit tests for the pure decision logic backing camera stall detection
    //! and reconnect gating. `maybe_reconnect_camera` itself opens a real
    //! camera device past its `should_reconnect` guard, so it's not
    //! unit-tested directly; the guard logic is tested here instead.
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

    // --- should_reconnect ---

    #[test]
    fn should_reconnect_false_when_no_stall_tracked() {
        let now = Instant::now();
        assert!(!should_reconnect(None, None, now));
    }

    #[test]
    fn should_reconnect_false_below_reconnect_stall_threshold() {
        let now = Instant::now();
        let seen = FrameLiveness {
            timestamp: now,
            unchanged_since: now - Duration::from_secs(5),
            warned: false,
        };
        assert!(!should_reconnect(Some(&seen), None, now));
    }

    #[test]
    fn should_reconnect_true_past_stall_threshold_with_no_prior_attempt() {
        let now = Instant::now();
        let seen = FrameLiveness {
            timestamp: now,
            unchanged_since: now - CAMERA_RECONNECT_STALL - Duration::from_secs(1),
            warned: false,
        };
        assert!(should_reconnect(Some(&seen), None, now));
    }

    #[test]
    fn should_reconnect_false_past_stall_threshold_but_within_cooldown() {
        let now = Instant::now();
        let seen = FrameLiveness {
            timestamp: now,
            unchanged_since: now - CAMERA_RECONNECT_STALL - Duration::from_secs(1),
            warned: false,
        };
        let last_attempt = now - Duration::from_secs(1);
        assert!(!should_reconnect(Some(&seen), Some(last_attempt), now));
    }

    #[test]
    fn should_reconnect_true_past_stall_threshold_and_past_cooldown() {
        let now = Instant::now();
        let seen = FrameLiveness {
            timestamp: now,
            unchanged_since: now - CAMERA_RECONNECT_STALL - Duration::from_secs(1),
            warned: false,
        };
        let last_attempt = now - CAMERA_RECONNECT_COOLDOWN - Duration::from_secs(1);
        assert!(should_reconnect(Some(&seen), Some(last_attempt), now));
    }

    // --- pre_buffer_ready ---

    #[test]
    fn pre_buffer_ready_true_when_no_reconnect_tracked() {
        let now = Instant::now();
        assert!(pre_buffer_ready(None, 10, now));
    }

    #[test]
    fn pre_buffer_ready_false_before_pre_buffer_secs_have_elapsed() {
        let now = Instant::now();
        let reconnected_at = now - Duration::from_secs(4);
        assert!(!pre_buffer_ready(Some(reconnected_at), 10, now));
    }

    #[test]
    fn pre_buffer_ready_true_once_pre_buffer_secs_have_elapsed() {
        let now = Instant::now();
        let reconnected_at = now - Duration::from_secs(10);
        assert!(pre_buffer_ready(Some(reconnected_at), 10, now));
    }

    #[test]
    fn pre_buffer_ready_true_immediately_when_pre_buffer_secs_is_zero() {
        let now = Instant::now();
        assert!(pre_buffer_ready(Some(now), 0, now));
    }
}
