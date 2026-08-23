use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::buffer::RingBuffer;
use crate::capture;
use crate::liveness::FrameLiveness;

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

#[cfg(test)]
mod tests {
    //! Unit tests for the pure decision logic backing reconnect gating.
    //! `maybe_reconnect_camera` itself opens a real camera device past its
    //! `should_reconnect` guard, so it's not unit-tested directly; the guard
    //! logic is tested here instead.
    #![allow(
        clippy::unwrap_used,
        clippy::missing_panics_doc,
        clippy::unchecked_time_subtraction,
        reason = "test assertions favor unwrap/plain time arithmetic for clarity; test durations \
                   are small hardcoded constants, so underflow is not reachable"
    )]

    use std::time::{Duration, Instant};

    use super::*;

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
}
