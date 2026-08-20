//! motioncap: webcam-based security motion capture. See the project guidance
//! file and `docs/adr/` for architecture and design-decision context.

/// Rolling pre-buffer of recent frames/audio (see `RingBuffer`).
mod buffer;
/// Camera and audio capture callbacks.
mod capture;
/// CLI argument parsing.
mod config;
/// The one function in `config` that cannot be unit-tested (see that
/// module's doc comment).
mod config_coverage_excluded;
/// Top-level wiring and the functions that cannot be unit-tested (see that
/// module's doc comment).
mod coverage_excluded;
/// YOLO object-detection inference.
mod detect;
/// The parts of `detect` that require a real ONNX Runtime session and model
/// file, and cannot be unit-tested (see that module's doc comment).
mod detect_coverage_excluded;
/// Background-subtraction motion gate. Excluded from the coverage report for
/// now via the `coverage_excluded` filename match (see `docs/adr/0006-coverage-exclusions.md`).
mod motion_coverage_excluded;
/// Shared `OpenCV` conversion helpers.
mod opencv_utils;
/// Output file/folder naming.
mod paths;
/// Opt-in live preview window.
mod preview;
/// Recording lifecycle and ffmpeg-backed encoding.
mod recorder;
/// The handful of `recorder` expressions that cannot be unit-tested (see
/// that module's doc comment).
mod recorder_coverage_excluded;
/// Startup dependency checks.
mod startup;
/// YOLO-detection-to-trigger evaluation.
mod triggers;

#[cfg(test)]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
#[cfg(test)]
use std::thread;
use std::time::Duration;

use anyhow::Result;

use buffer::{RingBuffer, TimestampedAudio, TimestampedFrame};
use config::Config;
use detect_coverage_excluded::Detector;
use motion_coverage_excluded::MotionGate;
use paths::clip_path;
use recorder::{RecordingEvent, RecordingEventParams};

/// An event that's been started (ffmpeg spawned) but whose pre-event buffer
/// hasn't been written yet. Kept separate from `RecordingEvent` construction
/// (see `RecordingEvent::start`'s docs) so the detection loop never blocks on
/// writing dozens of pre-buffer frames. The writer thread seeds it as the
/// first thing it does once it sees a pending event, then it becomes a
/// normal actively-written event.
struct PendingEvent {
    /// The started recording (ffmpeg already spawned) awaiting its pre-buffer seed.
    event: RecordingEvent,
    /// Pre-trigger frames to seed into `event` once the writer thread picks it up.
    pre_frames: Vec<TimestampedFrame>,
    /// Pre-trigger audio to seed into `event` once the writer thread picks it up.
    pre_audio: Vec<TimestampedAudio>,
}

/// Shared state for the currently in-progress recording, if any. Starts as
/// `Pending` (ffmpeg spawned, pre-buffer not yet written) so the writer
/// thread can seed it without blocking whichever thread created it; once
/// seeded it becomes `Active` and receives normal steady-paced writes.
pub(crate) enum ActiveEvent {
    /// No recording is in progress.
    None,
    /// ffmpeg has been spawned but the pre-event buffer hasn't been seeded yet.
    Pending(PendingEvent),
    /// The event is seeded and receiving normal steady-paced writes.
    Active(RecordingEvent),
}

impl ActiveEvent {
    /// Whether any recording (pending or active) is currently in progress.
    const fn is_some(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// The active `RecordingEvent`, if one is seeded and receiving writes.
    const fn as_recording_mut(&mut self) -> Option<&mut RecordingEvent> {
        match self {
            Self::None | Self::Pending(_) => None,
            Self::Active(event) => Some(event),
        }
    }

    /// Takes the in-progress recording (if any), leaving `None` behind.
    fn take(&mut self) -> Option<RecordingEvent> {
        match std::mem::replace(self, Self::None) {
            Self::None => None,
            Self::Pending(pending) => Some(pending.event),
            Self::Active(event) => Some(event),
        }
    }
}

/// Signals when the writer thread has completed its post-shutdown
/// last-chance drain (see `run_recording_writer_loop`), so the detection
/// loop's shutdown path can block on it instead of finalizing the active
/// event prematurely. A condvar rather than a busy-polled flag, since this
/// is a one-shot handshake during shutdown, not a recurring cadence.
#[derive(Default)]
pub(crate) struct WriterDrained {
    /// Set to `true` once the writer thread's final drain has completed.
    done: Mutex<bool>,
    /// Notified when `done` is set, to wake `wait`'s blocked receiver.
    condvar: Condvar,
}

impl WriterDrained {
    /// Marks the final drain as complete and wakes any thread blocked in `wait`.
    pub(crate) fn signal(&self) {
        *self.done.lock().expect("writer-drained lock poisoned") = true;
        self.condvar.notify_one();
    }

    /// Blocks until `signal` has been called.
    pub(crate) fn wait(&self) {
        let guard = self.done.lock().expect("writer-drained lock poisoned");
        drop(
            self.condvar
                .wait_while(guard, |done| !*done)
                .expect("writer-drained lock poisoned"),
        );
    }
}

/// Finalizes whatever recording is in progress (if any) on shutdown.
///
/// Deliberately does *not* use `ActiveEvent::take`, which collapses
/// `Pending`/`Active` down to a bare `RecordingEvent`: a `Pending` event has
/// ffmpeg spawned but no frames/audio written yet. That only happens once
/// the writer thread picks it up (see `ActiveEvent`'s docs). Finishing it
/// unseeded would hand `finish` an empty video/audio pair, producing a
/// malformed mux instead of a usable (if short) clip, so it must be seeded
/// here first, same as the writer thread would have done.
pub(crate) fn finish_event_on_shutdown(
    active_event: &Mutex<ActiveEvent>,
    writer_drained: &WriterDrained,
) -> Result<()> {
    // The writer thread does its own last-chance drain on shutdown (see
    // `run_recording_writer_loop`) so trailing footage captured while this
    // thread was mid-inference still lands in the clip. Wait for that drain
    // to actually happen before taking/finishing the event. Otherwise this
    // thread can race the writer thread and finish the clip first, in which
    // case the writer's later drain finds `ActiveEvent::None` and silently
    // drops that trailing footage instead of writing it.
    writer_drained.wait();

    let taken = std::mem::replace(
        &mut *active_event.lock().expect("active event lock poisoned"),
        ActiveEvent::None,
    );

    let event = match taken {
        ActiveEvent::None => None,
        ActiveEvent::Pending(mut pending) => {
            if let Err(err) = pending.event.seed(&pending.pre_frames, &pending.pre_audio) {
                log::error!("failed to seed pre-buffer into new recording: {err:?}");
            }
            Some(pending.event)
        }
        ActiveEvent::Active(event) => Some(event),
    };

    if let Some(event) = event {
        event.finish()?;
        log::info!("recording closed on shutdown");
    }

    Ok(())
}

/// Motion-gate + YOLO evaluation cadence. Kept separate from the recording
/// frame rate below since inference cost doesn't scale down usefully at
/// higher polling rates. 15fps is plenty for deciding whether a subject is
/// still present.
const DETECTION_FRAME_RATE: u32 = 15;
/// Poll interval derived from `DETECTION_FRAME_RATE`.
pub(crate) const DETECTION_POLL_INTERVAL: Duration =
    Duration::from_millis(1000 / DETECTION_FRAME_RATE as u64);

/// Recorded video frame rate, used by the writer thread and the video
/// encoder. Measured (via traced ring-buffer frame timestamps under real
/// running conditions: all threads active, real RGB decode load) at ~18fps
/// average delivery for this camera, well short of the 50-65fps seen in
/// isolated capture-only testing. Polling faster than the camera actually
/// delivers just makes the writer re-write stale frames, which plays back as
/// stutter/perceived speed-up (measured ~42% duplicate frame writes at
/// 30fps vs. 0% at 15fps). 15fps is the safe ceiling until the writer tracks
/// per-frame identity to skip real duplicates.
const RECORDING_FRAME_RATE: u32 = 15;
// RECORDING_POLL_INTERVAL (derived from RECORDING_FRAME_RATE) and
// PREVIEW_FRAME_RATE/PREVIEW_POLL_INTERVAL now live in coverage_excluded.rs,
// the only place they're used, alongside run_recording_writer_loop and
// run_preview_loop.

/// Log file name written under `--output-dir` (see `coverage_excluded::init_logging`).
pub(crate) const LOG_FILE_NAME: &str = "motioncap.log";

/// How long a camera stall (see `FrameLiveness`) must persist before
/// `run_detection_loop` tears down and rebuilds the capture stream, rather
/// than continuing to wait for it to recover on its own.
///
/// Deliberately much longer than `recorder::MAX_FRAME_STALL` (1.5s): that
/// threshold exists to stop feeding stale frames into detection/recording
/// within a couple of seconds, which is far too trigger-happy to also gate
/// tearing down and reopening the OS camera handle. Doing that on every
/// brief stall would thrash the device and could itself induce more stalls.
/// This threshold instead assumes the camera is genuinely gone (see
/// `capture::camera_coverage_excluded::start_camera_capture`'s doc comment for why nokhwa
/// never recovers from this on its own) and a full stream rebuild is
/// warranted.
const CAMERA_RECONNECT_STALL: Duration = Duration::from_secs(15);

/// Minimum time between reconnect attempts once the camera is believed dead,
/// so a camera that fails to reopen (e.g. genuinely unplugged) doesn't get a
/// reopen attempt on every single detection poll while it's absent.
const CAMERA_RECONNECT_COOLDOWN: Duration = Duration::from_secs(10);

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
const PENDING_CONFIRMATION_WINDOW: Duration = Duration::from_secs(5);

/// Writes every log line to both the given file and stderr, since this
/// process runs long-lived and unattended. Stderr alone
/// is lost the moment the terminal/session that launched it goes away, but
/// keeping stderr too means interactive/`--preview` runs still see live
/// diagnostics without needing to tail the file.
pub(crate) struct TeeWriter {
    /// The persistent log file under `--output-dir`.
    file: std::fs::File,
}

impl TeeWriter {
    /// Wraps an already-opened log file for tee'd write/flush to both it and stderr.
    pub(crate) const fn new(file: std::fs::File) -> Self {
        Self { file }
    }
}

/// Runs both fallible sink operations unconditionally, never short-circuiting
/// with `?` after just the first, and propagates the first error, if any.
/// One sink failing (e.g. a full disk, or stderr closed under a supervisor)
/// must never silently suppress the other from being attempted.
fn both(a: std::io::Result<()>, b: std::io::Result<()>) -> std::io::Result<()> {
    a?;
    b?;
    Ok(())
}

impl std::io::Write for TeeWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        both(self.file.write_all(buf), std::io::stderr().write_all(buf))?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        both(self.file.flush(), std::io::stderr().flush())
    }
}

// init_logging lives in coverage_excluded.rs, not here: it calls
// env_logger::Builder::init(), which sets the process-global logger and
// can't be safely called from a test (see coverage_excluded.rs's module doc).

/// Entry point. All actual startup/wiring logic lives in `coverage_excluded::run`
/// (see that module's doc comment for why it's split out into its own
/// coverage-excluded file rather than living here).
fn main() -> Result<()> {
    coverage_excluded::run()
}

/// Seeds a `Pending` event's pre-buffer (promoting it to `Active`) and drains
/// any newly-captured frames/audio into whichever event is active, if any.
/// Shared between the writer thread's normal poll cadence and its
/// shutdown path (see `run_recording_writer_loop`), so a shutdown that lands
/// mid-inference in the detection loop still gets one last drain instead of
/// silently dropping trailing footage the writer thread would otherwise
/// never pick up.
///
/// The lock is deliberately held across the seed/drain calls below, not just
/// the state-swap: `active_event` must reflect "a recording is in flight"
/// continuously for the detection loop's shutdown path (which calls `take()`
/// and finishes the event) and its `is_some()` check to observe consistent
/// state. Releasing it mid-drain would open a window where a concurrent
/// shutdown fails to finalize the in-flight clip.
#[allow(
    clippy::significant_drop_tightening,
    reason = "guard must stay held across seed/drain so shutdown's take() can't race an in-flight event"
)]
fn seed_and_drain_active_event(ring_buffer: &Mutex<RingBuffer>, active_event: &Mutex<ActiveEvent>) {
    let mut guard = active_event.lock().expect("active event lock poisoned");

    let taken = std::mem::replace(&mut *guard, ActiveEvent::None);

    match taken {
        ActiveEvent::Pending(mut pending) => {
            if let Err(err) = pending.event.seed(&pending.pre_frames, &pending.pre_audio) {
                log::error!("failed to seed pre-buffer into new recording: {err:?}");
            }
            *guard = ActiveEvent::Active(pending.event);
        }
        other => *guard = other,
    }

    let Some(event) = guard.as_recording_mut() else {
        return;
    };

    if let Err(err) = event.drain_frames(ring_buffer) {
        log::error!("failed to drain frames into active recording: {err:?}");
    }

    if let Err(err) = event.drain_audio(ring_buffer) {
        log::error!("failed to drain audio into active recording: {err:?}");
    }
}

// run_recording_writer_loop and run_preview_loop live in coverage_excluded.rs,
// not here: past their shutdown-check branch (unit-tested there via a
// pre-set shutdown flag) both loops spend their steady-state body polling
// thread::sleep against real wall-clock time and, in run_preview_loop's
// case, driving OpenCV's highgui window (see coverage_excluded.rs's module
// doc), so neither is exercisable by a test without either running forever
// or a real display. run_recording_writer_loop's per-tick logic
// (seed_and_drain_active_event) is independently unit-tested here;
// run_preview_loop's per-tick logic (PreviewWindow::show) is excluded
// entirely, along with the rest of preview.rs, per ADR 6.

/// Closes the active recording if either close condition is met: the camera
/// has stalled (see `RecordingEvent::camera_stalled`), which the motion gate
/// can never detect on its own since a stalled camera means no new frames
/// ever reach it to evaluate; or the post-buffer quiet window has elapsed
/// with no fresh trigger. No-ops if neither condition holds.
///
/// Takes the `MutexGuard` by value so it can be dropped before `finish`
/// (which waits on ffmpeg) runs. The lock must not be held across that.
pub(crate) fn close_event_if_done(
    mut guard: MutexGuard<'_, ActiveEvent>,
    post_buffer: Duration,
) -> Result<()> {
    let Some(event) = guard.as_recording_mut() else {
        return Ok(());
    };

    let stalled = event.camera_stalled();
    let quiet_timed_out = event.quiet_for() >= post_buffer;

    if stalled || quiet_timed_out {
        let event = guard.take().expect("checked Some above");
        drop(guard);
        event.finish()?;

        if stalled {
            log::warn!("recording closed: camera stopped delivering frames");
        } else {
            log::info!("recording closed");
        }
    }

    Ok(())
}

/// Tracks the most recent frame timestamp `run_detection_loop` has evaluated,
/// and how long that timestamp has been unchanged, to detect a stalled camera
/// before any recording has started (see `frame_liveness_advanced`).
pub(crate) struct FrameLiveness {
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

/// A first, not-yet-confirmed living-thing detection while no recording is
/// active, awaiting a second sighting of the same class within
/// `PENDING_CONFIRMATION_WINDOW` before `run_detection_loop` will actually
/// start a recording (see that constant's docs for why single-poll
/// confirmation isn't trustworthy on its own).
pub(crate) struct PendingConfirmation {
    /// The living-thing class seen on the first, unconfirmed poll.
    class_name: &'static str,
    /// When `class_name` was last seen: the first, unconfirmed poll, or (once
    /// confirmed) the most recent poll that re-confirmed it. See
    /// `confirm_pending`'s doc comment for why this refreshes on every
    /// confirmed repeat rather than staying fixed at the first sighting.
    first_seen: std::time::Instant,
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
fn expire_stale_pending(pending: &mut Option<PendingConfirmation>, now: std::time::Instant) {
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
fn confirm_pending(
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
fn poll_confirmed_detections(
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

/// Updates `last_seen` for a newly-polled `latest_frame` timestamp and
/// reports whether the loop should proceed with it. Returns `false` once the
/// same timestamp has recurred for `recorder::MAX_FRAME_STALL`, i.e.
/// `latest_frame()` is returning a frame the camera delivered a while ago,
/// not a fresh one, which otherwise would get silently re-run through
/// motion/YOLO on every poll and cascade into duplicate recordings. A
/// same-timestamp recurrence shorter than that is ordinary jitter between
/// polls and camera delivery, not a stall, so it's allowed through unlogged.
/// The stall is logged once (not once per poll) when it's first detected.
pub(crate) fn frame_liveness_advanced(
    last_seen: &mut Option<FrameLiveness>,
    frame_timestamp: std::time::Instant,
) -> bool {
    let now = std::time::Instant::now();

    match last_seen {
        Some(seen) if seen.timestamp == frame_timestamp => {
            if seen.stalled_for(now) < recorder::MAX_FRAME_STALL {
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

// maybe_reconnect_camera lives in coverage_excluded.rs, not here: past the
// should_reconnect guard below (which is what's actually unit-tested),
// every remaining line calls capture::camera_coverage_excluded::start_camera_capture, which
// opens a real /dev/videoN device (see coverage_excluded.rs's module doc). The
// pure threshold/cooldown decision is should_reconnect, kept here and
// tested directly.

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
fn pre_buffer_ready(
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
pub(crate) fn reset_liveness_after_reconnect(last_seen: &mut Option<FrameLiveness>) {
    if let Some(seen) = last_seen {
        seen.unchanged_since = std::time::Instant::now();
        seen.warned = false;
    }
}

/// Runs the motion gate and (on trip) YOLO confirmation against `frame` for
/// an already-active recording, records the result into its sidecar, and
/// closes the event if either close condition in `close_event_if_done` is
/// met. Kept separate from `run_detection_loop`'s no-active-event path since
/// the two have no logic in common beyond polling the same frame.
///
/// A YOLO hit here goes through the same `confirm_pending` repeat-sighting
/// gate as starting a new recording (see `PENDING_CONFIRMATION_WINDOW`),
/// rather than being trusted on the first poll: an already-active recording
/// doesn't make its own single-frame hits any more trustworthy, and without
/// this gate a scene that keeps producing recurring (not just one-off)
/// hallucinations (observed directly: the same misclassified class
/// recurring for minutes on an empty room once *something* had genuinely
/// triggered the recording earlier) can keep re-extending the post-buffer
/// window on noise alone long after the real subject has left frame.
///
/// Separately, a bare motion-gate trip with no YOLO confirmation at all only
/// extends the post-buffer window while `pending_confirmation` is `Some`,
/// i.e. a class was seen recently, confirmed or not, never unconditionally.
/// Per ADR 2, "correctness against false positives comes entirely from the
/// YOLO confirmation requirement"; trusting motion alone here (the original
/// behavior) let ordinary sensor jitter well below any living-thing
/// detection (observed directly: 100+ sub-threshold motion trips with zero
/// confirmed detections anywhere in that stretch) keep a clip open for
/// minutes after the confirmed subject had actually left frame.
#[allow(
    clippy::significant_drop_tightening,
    reason = "guard is moved into close_event_if_done, which drops it itself before the slow finish() call"
)]
#[allow(
    clippy::too_many_arguments,
    reason = "each parameter is independently threaded state from run_detection_loop's loop body, not a natural grouping (matches try_start_recording's identical justification)"
)]
pub(crate) fn evaluate_active_event(
    config: &Config,
    motion_gate: &mut MotionGate,
    detector: &mut Detector,
    active_event: &Arc<Mutex<ActiveEvent>>,
    frame: &image::RgbImage,
    frame_timestamp: std::time::Instant,
    post_buffer: Duration,
    pending_confirmation: &mut Option<PendingConfirmation>,
) -> Result<()> {
    let motion = motion_gate.evaluate(frame)?;

    // `detector.detect` runs without holding the event lock since YOLO
    // inference is the slow step; the recording writer thread must be free
    // to keep writing frames/audio at a steady pace while this runs, not
    // blocked waiting on this lock.
    let confirmed = if motion.tripped {
        poll_confirmed_detections(detector, config, frame, pending_confirmation)?
    } else {
        None
    };

    let mut guard = active_event.lock().expect("active event lock poisoned");
    let Some(event) = guard.as_recording_mut() else {
        // Recording was closed elsewhere (e.g. shutdown) while inference was
        // running above, or the writer thread hasn't finished seeding it yet.
        return Ok(());
    };

    if motion.tripped {
        event.record_motion(motion.changed_ratio, frame_timestamp);

        if let Some(confirmed) = &confirmed {
            for d in confirmed {
                event.record_detection(d.class_name, d.confidence, frame_timestamp);
            }
        } else if pending_confirmation.is_some() {
            // Motion continues but wasn't re-confirmed by YOLO this frame;
            // still reset the quiet-window so a subject that briefly stops
            // moving doesn't get cut off early. Gated on `pending_confirmation`
            // being `Some` (a class was seen recently, confirmed or not)
            // rather than unconditional, so bare motion-gate noise with no
            // recent YOLO sighting at all can't extend the window on its own.
            // See this function's doc comment for the observed failure
            // mode this prevents.
            event.touch();
        }
    }

    close_event_if_done(guard, post_buffer)
}

/// The shared camera handle `run_detection_loop` polls liveness against and,
/// on a prolonged stall, rebuilds via `maybe_reconnect_camera`. Bundled
/// together (rather than passed as two separate `run_detection_loop`
/// parameters) both because they're always used as a pair at that call site
/// and because `run_detection_loop` is already at clippy's argument-count
/// limit.
pub(crate) struct DetectionCamera {
    /// The live capture stream; swapped out in place on reconnect. `None`
    /// only for the brief window between dropping the old stream and
    /// successfully opening the replacement. See `maybe_reconnect_camera`.
    pub(crate) handle: Arc<Mutex<Option<nokhwa::threaded::CallbackCamera>>>,
    /// The originally configured device (or `None` for auto-detect), reused
    /// unchanged on every reconnect attempt.
    pub(crate) device: Option<std::path::PathBuf>,
}

/// Audio stream parameters the recording writer needs to configure ffmpeg's
/// input, captured once at startup and passed through unchanged.
pub(crate) struct AudioParams {
    /// Sample rate of the captured audio stream, in Hz.
    pub(crate) sample_rate: u32,
    /// Number of audio channels in the captured stream.
    pub(crate) channels: u16,
}

// run_detection_loop lives in coverage_excluded.rs, not here: it constructs a
// real Detector::load unconditionally before its loop body ever runs, so even
// its shutdown-only path can't be unit-tested without the model file/ONNX
// Runtime dependency (see coverage_excluded.rs's module doc). Every per-tick
// decision it makes is independently covered here via the functions it calls
// (try_start_recording, evaluate_active_event, frame_liveness_advanced,
// maybe_reconnect_camera, finish_event_on_shutdown, etc.).

/// Runs the motion gate and (on trip, then second-poll confirmation) YOLO
/// detection against `frame` when no recording is currently active, starting
/// a new `ActiveEvent::Pending` once `confirm_pending` accepts a repeat
/// sighting and `pre_buffer_ready` confirms the ring buffer has had time to
/// refill since the last camera reconnect (see that function's doc comment).
/// Kept separate from `run_detection_loop` purely to stay under clippy's
/// function-length limit; it has no logic in common with
/// `evaluate_active_event` beyond polling the same frame.
#[allow(
    clippy::too_many_arguments,
    reason = "each parameter is independently threaded state from run_detection_loop's loop body, not a natural grouping"
)]
pub(crate) fn try_start_recording(
    config: &Config,
    motion_gate: &mut MotionGate,
    detector: &mut Detector,
    ring_buffer: &Arc<Mutex<RingBuffer>>,
    active_event: &Arc<Mutex<ActiveEvent>>,
    audio: &AudioParams,
    frame: &image::RgbImage,
    frame_timestamp: std::time::Instant,
    pending_confirmation: &mut Option<PendingConfirmation>,
    reconnected_at: Option<std::time::Instant>,
) -> Result<()> {
    let motion = motion_gate.evaluate(frame)?;

    log::trace!("frame received; motion_tripped={}", motion.tripped);

    if !motion.tripped {
        return Ok(());
    }

    if !pre_buffer_ready(reconnected_at, config.pre_buffer_secs, frame_timestamp) {
        // The capture stream was rebuilt too recently for the ring buffer to
        // have refilled a full pre-buffer window; skip the expensive YOLO
        // call entirely rather than running inference on every tripped-motion
        // poll for the whole grace window, only to discard the result (see
        // `pre_buffer_ready`). `expire_stale_pending` is still called
        // directly (bypassing `poll_confirmed_detections`, which only runs
        // it as a side effect of an inference call this path deliberately
        // skips) so a `pending_confirmation` older than
        // `PENDING_CONFIRMATION_WINDOW` can't sit unexpired for the whole
        // grace window and then spuriously confirm against an unrelated
        // sighting once inference resumes. `pending_confirmation` is
        // otherwise left untouched so a still-present subject simply
        // re-confirms and retries on the next poll instead of needing a
        // fresh two-poll confirmation cycle once the buffer is ready.
        expire_stale_pending(pending_confirmation, frame_timestamp);
        log::debug!(
            "motion tripped but recording start held back; ring buffer still refilling after \
             camera reconnect"
        );
        return Ok(());
    }

    let Some(confirmed) = poll_confirmed_detections(detector, config, frame, pending_confirmation)?
    else {
        return Ok(());
    };

    let mut classes: Vec<&str> = confirmed.iter().map(|d| d.class_name).collect();

    classes.sort_unstable();
    classes.dedup();

    let (pre_frames, pre_audio) = {
        let buf = ring_buffer.lock().expect("ring buffer lock poisoned");
        buf.snapshot()
    };

    let Some(first_pre_frame) = pre_frames.first() else {
        // No buffered frames yet (e.g. trigger fired immediately at startup,
        // before the camera has produced anything); skip this trigger rather
        // than starting a recording with no video.
        return Ok(());
    };

    let (width, height) = first_pre_frame.image.dimensions();
    let clip_timeline_start = first_pre_frame.timestamp;

    let started_at = chrono::Local::now();
    let path = clip_path(&config.output_dir, started_at, &classes)?;
    let mut event = RecordingEvent::start(RecordingEventParams {
        final_clip_path: path,
        output_dir: config.output_dir.clone(),
        started_at,
        width,
        height,
        frame_rate: RECORDING_FRAME_RATE,
        audio_sample_rate: audio.sample_rate,
        audio_channels: audio.channels,
        clip_timeline_start,
    })?;

    event.record_motion(motion.changed_ratio, frame_timestamp);

    for d in &confirmed {
        event.record_detection(d.class_name, d.confidence, frame_timestamp);
    }

    log::info!("recording started: {classes:?}");

    *active_event.lock().expect("active event lock poisoned") =
        ActiveEvent::Pending(PendingEvent {
            event,
            pre_frames,
            pre_audio,
        });

    // Clear rather than leave populated: once this recording closes (e.g. a
    // short post-buffer or a camera stall), the loop falls back to this same
    // function with a fresh trigger. A leftover `pending_confirmation` from
    // the sighting that just started *this* recording could otherwise let
    // that unrelated next trigger pass the repeat-sighting gate on its very
    // first poll if it happens to land within the window, exactly the
    // stale-state hazard `active_pending_confirmation` is kept separate from
    // `pending_confirmation` to avoid on the active-event side.
    *pending_confirmation = None;

    Ok(())
}

#[cfg(test)]
mod tests {
    //! Unit tests for the pure decision logic backing the detection worker's
    //! repeat-sighting confirmation gate, stall detection, and shutdown
    //! handshake. `ActiveEvent`'s `Pending`/`Active` variants (which wrap a
    //! real `RecordingEvent`, itself requiring a spawned ffmpeg process) are
    //! left to the `ClipState` refactor's own tests, which can construct one
    //! without ffmpeg.
    #![allow(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        clippy::missing_panics_doc,
        clippy::unchecked_time_subtraction,
        reason = "test assertions favor unwrap/indexing/plain time arithmetic for clarity; test \
                   durations are small hardcoded constants, so underflow is not reachable"
    )]

    use std::io;
    use std::time::Instant;

    use clap::Parser as _;
    use detect::Detection;

    use super::*;

    /// Pushes one fresh frame through `event.drain_frames`, refreshing its
    /// `last_real_frame_at` (the `camera_stalled` clock) to "now". Needed
    /// immediately before any assertion that an event stays open despite
    /// `MAX_FRAME_STALL` being only 1.5s: under a parallel `cargo test` run
    /// with many concurrent ffmpeg/OpenCV/ONNX-Runtime-backed tests, more
    /// than 1.5s can genuinely elapse between `test_recording_event`
    /// constructing (and seeding) an event and a later test actually calling
    /// `evaluate_active_event` against it; confirmed directly: a test using
    /// a freshly-seeded event without this refresh passed reliably alone but
    /// flaked under `cargo test`'s default full-suite parallelism.
    fn refresh_event_liveness(event: &mut RecordingEvent) {
        let ring_buffer = Mutex::new(RingBuffer::new(Duration::from_secs(10)));
        ring_buffer
            .lock()
            .unwrap()
            .push_frame(image::RgbImage::new(2, 2));
        event.drain_frames(&ring_buffer).unwrap();
    }

    fn detection(class_name: &'static str) -> Detection {
        Detection {
            class_name,
            confidence: 0.9,
        }
    }

    /// Starts a real (ffmpeg-backed) 2x2 `RecordingEvent` in `dir` and seeds
    /// it with one frame, for tests that need an actual `ActiveEvent::Active`
    /// rather than `None`.
    ///
    /// Seeding at least one frame is required, not cosmetic: an event
    /// `finish()`ed with zero video frames written produces a `Duration: N/A`
    /// video stream, and `mux_audio_into_video`'s `apad` filter has no
    /// duration to pad audio *to* in that case; `-shortest` never trips,
    /// so ffmpeg pads forever and the mux process hangs indefinitely
    /// (confirmed directly: reproduced standalone with a zero-frame video and
    /// empty audio file, `apad` ran for 8+ seconds generating unbounded
    /// output before being killed). A real recording always has pre-buffer
    /// frames seeded before anything can call `finish()`, so this matches
    /// production usage, not just what makes the test terminate.
    fn test_recording_event(dir: &std::path::Path) -> RecordingEvent {
        let clip_timeline_start = Instant::now();
        let started_at = chrono::Local::now();
        let path = clip_path(dir, started_at, &[]).unwrap();

        let mut event = RecordingEvent::start(RecordingEventParams {
            final_clip_path: path,
            output_dir: dir.to_path_buf(),
            started_at,
            width: 2,
            height: 2,
            frame_rate: 5,
            audio_sample_rate: 8000,
            audio_channels: 1,
            clip_timeline_start,
        })
        .unwrap();

        event
            .seed(
                &[TimestampedFrame {
                    timestamp: clip_timeline_start,
                    image: image::RgbImage::new(2, 2),
                }],
                &[],
            )
            .unwrap();

        event
    }

    // --- ActiveEvent::None behavior ---

    #[test]
    fn active_event_none_is_not_some() {
        assert!(!ActiveEvent::None.is_some());
    }

    #[test]
    fn active_event_none_has_no_recording_mut() {
        let mut event = ActiveEvent::None;
        assert!(event.as_recording_mut().is_none());
    }

    #[test]
    fn active_event_none_take_returns_none() {
        let mut event = ActiveEvent::None;
        assert!(event.take().is_none());
    }

    #[test]
    fn active_event_active_is_some_and_has_recording_mut() {
        let dir = tempfile::tempdir().unwrap();
        let mut event = ActiveEvent::Active(test_recording_event(dir.path()));

        assert!(event.is_some());
        assert!(event.as_recording_mut().is_some());

        event.take().unwrap().finish().unwrap();
    }

    #[test]
    fn active_event_pending_is_some_but_has_no_recording_mut() {
        let dir = tempfile::tempdir().unwrap();
        let mut event = ActiveEvent::Pending(PendingEvent {
            event: test_recording_event(dir.path()),
            pre_frames: Vec::new(),
            pre_audio: Vec::new(),
        });

        // Pending has ffmpeg spawned but no pre-buffer seeded yet, so
        // as_recording_mut deliberately withholds it; writing to an
        // unseeded event would happen before the writer thread's seed().
        assert!(event.is_some());
        assert!(event.as_recording_mut().is_none());

        event.take().unwrap().finish().unwrap();
    }

    #[test]
    fn active_event_take_collapses_pending_and_active_to_the_wrapped_event() {
        let dir = tempfile::tempdir().unwrap();
        let mut pending = ActiveEvent::Pending(PendingEvent {
            event: test_recording_event(dir.path()),
            pre_frames: Vec::new(),
            pre_audio: Vec::new(),
        });
        assert!(pending.take().is_some());
        assert!(matches!(pending, ActiveEvent::None));

        let mut active = ActiveEvent::Active(test_recording_event(dir.path()));
        let taken = active.take();
        assert!(taken.is_some());
        assert!(matches!(active, ActiveEvent::None));
        taken.unwrap().finish().unwrap();
    }

    // --- close_event_if_done ---

    #[test]
    fn close_event_if_done_noop_when_neither_condition_met() {
        let dir = tempfile::tempdir().unwrap();
        let event = Mutex::new(ActiveEvent::Active(test_recording_event(dir.path())));

        close_event_if_done(event.lock().unwrap(), Duration::from_mins(1)).unwrap();

        // Still active: neither camera_stalled nor the (60s) post-buffer
        // window has elapsed on a freshly-started event.
        assert!(event.lock().unwrap().is_some());
        event.lock().unwrap().take().unwrap().finish().unwrap();
    }

    #[test]
    fn close_event_if_done_closes_on_quiet_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let event = Mutex::new(ActiveEvent::Active(test_recording_event(dir.path())));

        // A post_buffer of 0 means quiet_for() >= post_buffer immediately.
        close_event_if_done(event.lock().unwrap(), Duration::ZERO).unwrap();

        assert!(!event.lock().unwrap().is_some());
    }

    #[test]
    fn close_event_if_done_closes_on_camera_stall() {
        let dir = tempfile::tempdir().unwrap();
        let event = Mutex::new(ActiveEvent::Active(test_recording_event(dir.path())));

        // A generous post_buffer means only the camera-stall branch (not the
        // quiet-window timeout) can be what closes this event. Backdated
        // rather than a real sleep, per ADR 7 (see
        // `backdate_last_real_frame_at_past_stall`'s doc comment).
        event
            .lock()
            .unwrap()
            .as_recording_mut()
            .unwrap()
            .state
            .backdate_last_real_frame_at_past_stall();

        close_event_if_done(event.lock().unwrap(), Duration::from_mins(1)).unwrap();

        assert!(!event.lock().unwrap().is_some());
    }

    #[test]
    fn close_event_if_done_is_noop_on_pending_or_none() {
        let dir = tempfile::tempdir().unwrap();
        let event = Mutex::new(ActiveEvent::Pending(PendingEvent {
            event: test_recording_event(dir.path()),
            pre_frames: Vec::new(),
            pre_audio: Vec::new(),
        }));

        // as_recording_mut() returns None for Pending, so this must be a
        // no-op even with a zero post_buffer; the event stays Pending.
        close_event_if_done(event.lock().unwrap(), Duration::ZERO).unwrap();
        assert!(event.lock().unwrap().is_some());
        event.lock().unwrap().take().unwrap().finish().unwrap();

        let none_event = Mutex::new(ActiveEvent::None);
        close_event_if_done(none_event.lock().unwrap(), Duration::ZERO).unwrap();
        assert!(!none_event.lock().unwrap().is_some());
    }

    // --- finish_event_on_shutdown ---

    #[test]
    fn finish_event_on_shutdown_noop_on_none() {
        let active_event = Mutex::new(ActiveEvent::None);
        let writer_drained = WriterDrained::default();
        writer_drained.signal();

        finish_event_on_shutdown(&active_event, &writer_drained).unwrap();

        assert!(!active_event.lock().unwrap().is_some());
    }

    #[test]
    fn finish_event_on_shutdown_finishes_an_active_event() {
        let dir = tempfile::tempdir().unwrap();
        let started_at = chrono::Local::now();
        let expected_path = clip_path(dir.path(), started_at, &[]).unwrap();

        let active_event = Mutex::new(ActiveEvent::Active(test_recording_event(dir.path())));
        let writer_drained = WriterDrained::default();
        writer_drained.signal();

        finish_event_on_shutdown(&active_event, &writer_drained).unwrap();

        assert!(!active_event.lock().unwrap().is_some());
        assert!(expected_path.exists());
    }

    #[test]
    fn finish_event_on_shutdown_seeds_then_finishes_a_pending_event() {
        let dir = tempfile::tempdir().unwrap();
        let started_at = chrono::Local::now();
        let expected_path = clip_path(dir.path(), started_at, &[]).unwrap();

        let active_event = Mutex::new(ActiveEvent::Pending(PendingEvent {
            event: test_recording_event(dir.path()),
            pre_frames: Vec::new(),
            pre_audio: Vec::new(),
        }));
        let writer_drained = WriterDrained::default();
        writer_drained.signal();

        // Pending must be seeded (even with an empty pre-buffer) before
        // finish(), or finish() would mux an empty video/audio pair.
        finish_event_on_shutdown(&active_event, &writer_drained).unwrap();

        assert!(!active_event.lock().unwrap().is_some());
        assert!(expected_path.exists());
    }

    // --- seed_and_drain_active_event ---

    #[test]
    fn seed_and_drain_active_event_seeds_a_pending_event_into_active() {
        let dir = tempfile::tempdir().unwrap();
        let clip_timeline_start = Instant::now();
        let started_at = chrono::Local::now();
        let path = clip_path(dir.path(), started_at, &[]).unwrap();

        let event = RecordingEvent::start(RecordingEventParams {
            final_clip_path: path,
            output_dir: dir.path().to_path_buf(),
            started_at,
            width: 2,
            height: 2,
            frame_rate: 5,
            audio_sample_rate: 8000,
            audio_channels: 1,
            clip_timeline_start,
        })
        .unwrap();

        let active_event = Mutex::new(ActiveEvent::Pending(PendingEvent {
            event,
            pre_frames: vec![TimestampedFrame {
                timestamp: clip_timeline_start,
                image: image::RgbImage::new(2, 2),
            }],
            pre_audio: Vec::new(),
        }));
        let ring_buffer = Mutex::new(RingBuffer::new(Duration::from_secs(10)));

        seed_and_drain_active_event(&ring_buffer, &active_event);

        // Pending -> Active: as_recording_mut() only returns Some once seeded.
        assert!(active_event.lock().unwrap().as_recording_mut().is_some());
        active_event
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .finish()
            .unwrap();
    }

    #[test]
    fn seed_and_drain_active_event_drains_new_frames_into_an_already_active_event() {
        let dir = tempfile::tempdir().unwrap();
        let active_event = Mutex::new(ActiveEvent::Active(test_recording_event(dir.path())));
        let ring_buffer = Mutex::new(RingBuffer::new(Duration::from_secs(10)));
        ring_buffer
            .lock()
            .unwrap()
            .push_frame(image::RgbImage::new(2, 2));

        // Must not panic/error against a genuinely Active (not Pending) event.
        seed_and_drain_active_event(&ring_buffer, &active_event);

        assert!(active_event.lock().unwrap().is_some());
        active_event
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .finish()
            .unwrap();
    }

    #[test]
    fn seed_and_drain_active_event_is_noop_on_none() {
        let active_event = Mutex::new(ActiveEvent::None);
        let ring_buffer = Mutex::new(RingBuffer::new(Duration::from_secs(10)));

        seed_and_drain_active_event(&ring_buffer, &active_event);

        assert!(!active_event.lock().unwrap().is_some());
    }

    // maybe_reconnect_camera now lives in coverage_excluded.rs (it opens a real
    // camera device past its should_reconnect guard); the guard logic itself
    // is tested directly below, under "should_reconnect".

    // --- WriterDrained ---

    #[test]
    fn writer_drained_signal_then_wait_returns_immediately() {
        let drained = WriterDrained::default();
        drained.signal();
        drained.wait(); // must not block
    }

    #[test]
    fn writer_drained_wait_blocks_until_signaled() {
        let drained = Arc::new(WriterDrained::default());
        let signaler = Arc::clone(&drained);

        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            signaler.signal();
        });

        drained.wait();
        handle.join().unwrap();
    }

    // --- TeeWriter ---
    //
    // init_logging itself is not unit-tested: it calls env_logger::Builder::init(),
    // which sets the process-global logger and panics if called more than once --
    // since cargo test runs the whole suite in one process, exercising it here
    // would either poison every other test's logging or only be safely callable
    // once, in an order-dependent way. TeeWriter's actual read/write logic (the
    // part `both` doesn't already cover) is tested directly below instead,
    // without going through init_logging's global side effect.

    #[test]
    fn tee_writer_write_appends_to_file_and_returns_full_length() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tee.log");
        let file = std::fs::File::create(&path).unwrap();
        let mut tee = TeeWriter::new(file);

        let n = std::io::Write::write(&mut tee, b"hello\n").unwrap();

        assert_eq!(n, 6);
        let contents = std::fs::read_to_string(&path).unwrap();
        assert_eq!(contents, "hello\n");
    }

    #[test]
    fn tee_writer_flush_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tee.log");
        let file = std::fs::File::create(&path).unwrap();
        let mut tee = TeeWriter::new(file);

        std::io::Write::write_all(&mut tee, b"data").unwrap();
        assert!(std::io::Write::flush(&mut tee).is_ok());
    }

    // --- both ---

    #[test]
    fn both_returns_ok_when_both_succeed() {
        assert!(both(Ok(()), Ok(())).is_ok());
    }

    #[test]
    fn both_returns_first_error_when_first_fails() {
        let err = both(
            Err(io::Error::other("first")),
            Err(io::Error::other("second")),
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "first");
    }

    #[test]
    fn both_returns_second_error_when_only_second_fails() {
        let err = both(Ok(()), Err(io::Error::other("second"))).unwrap_err();
        assert_eq!(err.to_string(), "second");
    }

    #[test]
    fn both_attempts_second_even_when_first_fails() {
        // The second closure's side effect (incrementing this counter) must
        // still run even though the first result is already an error --
        // `both` must not short-circuit before attempting both sinks.
        let attempted = Arc::new(AtomicBool::new(false));
        let attempted_clone = Arc::clone(&attempted);

        let second_result: io::Result<()> = {
            attempted_clone.store(true, Ordering::SeqCst);
            Ok(())
        };

        let _ = both(Err(io::Error::other("first")), second_result);
        assert!(attempted.load(Ordering::SeqCst));
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
        // recorder::MAX_FRAME_STALL is a hardcoded const (1.5s), not
        // injectable, so this test genuinely waits it out rather than
        // asserting the boundary logic through a shortened duration.
        let ts = Instant::now();
        let mut last_seen = Some(FrameLiveness {
            timestamp: ts,
            unchanged_since: Instant::now() - recorder::MAX_FRAME_STALL - Duration::from_millis(50),
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

    // run_recording_writer_loop and run_preview_loop, and their
    // shutdown-path tests, now live in coverage_excluded.rs alongside
    // run_detection_loop, since past their shutdown check both spend their
    // steady-state body on real wall-clock polling (and, for the preview
    // loop, a real OpenCV window) that can't be exercised here; see that
    // module's doc comment.

    // --- evaluate_active_event / try_start_recording (real Detector + MotionGate) ---
    //
    // Both functions call detector.detect on every motion-gate trip, so
    // meaningfully exercising anything past "motion didn't trip" requires a
    // real Detector::load (model file + ONNX Runtime), same as detect.rs's
    // #[ignore]'d tests; these are #[ignore]'d for the same reason and run
    // locally via `cargo test -- --ignored`. A synthetic changed-region frame
    // has no real living-thing subject in it, so poll_confirmed_detections
    // reliably returns None here even when motion trips; that's sufficient to
    // cover every branch except "a confirmed detection actually starts/
    // extends a recording", which would need a real photo of a person/animal
    // to exercise honestly rather than a synthetic frame.

    fn test_config(output_dir: &std::path::Path) -> Config {
        Config::try_parse_from([
            "motioncap",
            "--output-dir",
            &output_dir.to_string_lossy(),
            "--force-cpu",
        ])
        .unwrap()
    }

    fn test_detector() -> Detector {
        Detector::load(std::path::Path::new("models/yolov8n.onnx"), true)
            .expect("failed to load model, is models/yolov8n.onnx present?")
    }

    /// A 64x64 solid-color frame, for warming up `MotionGate`'s background model.
    fn background_frame() -> image::RgbImage {
        image::RgbImage::from_pixel(64, 64, image::Rgb([50, 50, 50]))
    }

    /// Same dimensions as `background_frame` but with a large changed region,
    /// reliably tripping a `MotionGate` already warmed up on the background.
    fn changed_frame() -> image::RgbImage {
        let mut frame = background_frame();
        for y in 0..32 {
            for x in 0..32 {
                frame.put_pixel(x, y, image::Rgb([250, 250, 250]));
            }
        }
        frame
    }

    #[test]
    #[ignore = "requires models/yolov8n.onnx + a working ONNX Runtime build; run explicitly with \
                `cargo test -- --ignored` on a machine that has both"]
    fn evaluate_active_event_returns_ok_when_event_already_closed() {
        // "Recording was closed elsewhere": guard.as_recording_mut()
        // returns None for ActiveEvent::None, same as it would for a
        // shutdown that raced this call.
        let _guard = detect::MODEL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());
        let mut motion_gate = MotionGate::new(config.motion_threshold).unwrap();
        let mut detector = test_detector();
        let active_event = Arc::new(Mutex::new(ActiveEvent::None));
        let mut pending = None;

        let result = evaluate_active_event(
            &config,
            &mut motion_gate,
            &mut detector,
            &active_event,
            &background_frame(),
            Instant::now(),
            Duration::from_mins(1),
            &mut pending,
        );

        assert!(result.is_ok());
        assert!(!active_event.lock().unwrap().is_some());
    }

    #[test]
    #[ignore = "requires models/yolov8n.onnx + a working ONNX Runtime build; run explicitly with \
                `cargo test -- --ignored` on a machine that has both"]
    fn evaluate_active_event_no_motion_leaves_event_untouched_and_open() {
        let _guard = detect::MODEL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());
        let mut motion_gate = MotionGate::new(config.motion_threshold).unwrap();
        let mut detector = test_detector();
        let active_event = Arc::new(Mutex::new(ActiveEvent::Active(test_recording_event(
            dir.path(),
        ))));
        let mut pending = None;

        for _ in 0..5 {
            motion_gate.evaluate(&background_frame()).unwrap();
        }

        if let Some(event) = active_event.lock().unwrap().as_recording_mut() {
            refresh_event_liveness(event);
        }

        evaluate_active_event(
            &config,
            &mut motion_gate,
            &mut detector,
            &active_event,
            &background_frame(),
            Instant::now(),
            Duration::from_mins(1),
            &mut pending,
        )
        .unwrap();

        assert!(active_event.lock().unwrap().is_some());
        active_event
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .finish()
            .unwrap();
    }

    #[test]
    #[ignore = "requires models/yolov8n.onnx + a working ONNX Runtime build; run explicitly with \
                `cargo test -- --ignored` on a machine that has both"]
    fn evaluate_active_event_motion_without_confirmation_records_motion_only() {
        let _guard = detect::MODEL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());
        let mut motion_gate = MotionGate::new(config.motion_threshold).unwrap();
        let mut detector = test_detector();
        let active_event = Arc::new(Mutex::new(ActiveEvent::Active(test_recording_event(
            dir.path(),
        ))));
        let mut pending = None;

        for _ in 0..5 {
            motion_gate.evaluate(&background_frame()).unwrap();
        }

        if let Some(event) = active_event.lock().unwrap().as_recording_mut() {
            refresh_event_liveness(event);
        }

        evaluate_active_event(
            &config,
            &mut motion_gate,
            &mut detector,
            &active_event,
            &changed_frame(),
            Instant::now(),
            Duration::from_mins(1),
            &mut pending,
        )
        .unwrap();

        // A synthetic frame has no real subject, so poll_confirmed_detections
        // returns None; the event stays open (60s post_buffer, not timed
        // out) but no detection was confirmed. Whether `pending` ends up
        // `Some`/`None` depends on the model's actual output on this
        // synthetic frame, which isn't deterministic across builds, so it
        // isn't asserted on here. Only that evaluate_active_event ran
        // without panicking and left the event open is checked.
        assert!(active_event.lock().unwrap().is_some());
        active_event
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .finish()
            .unwrap();
    }

    #[test]
    #[ignore = "requires models/yolov8n.onnx + a working ONNX Runtime build; run explicitly with \
                `cargo test -- --ignored` on a machine that has both"]
    fn evaluate_active_event_closes_when_quiet_window_elapsed() {
        let _guard = detect::MODEL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());
        let mut motion_gate = MotionGate::new(config.motion_threshold).unwrap();
        let mut detector = test_detector();
        let active_event = Arc::new(Mutex::new(ActiveEvent::Active(test_recording_event(
            dir.path(),
        ))));
        let mut pending = None;

        for _ in 0..5 {
            motion_gate.evaluate(&background_frame()).unwrap();
        }

        evaluate_active_event(
            &config,
            &mut motion_gate,
            &mut detector,
            &active_event,
            &background_frame(),
            Instant::now(),
            Duration::ZERO,
            &mut pending,
        )
        .unwrap();

        assert!(!active_event.lock().unwrap().is_some());
    }

    #[test]
    #[ignore = "requires models/yolov8n.onnx + a working ONNX Runtime build; run explicitly with \
                `cargo test -- --ignored` on a machine that has both"]
    fn try_start_recording_no_motion_does_not_start_a_recording() {
        let _guard = detect::MODEL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());
        let mut motion_gate = MotionGate::new(config.motion_threshold).unwrap();
        let mut detector = test_detector();
        let ring_buffer = Arc::new(Mutex::new(RingBuffer::new(Duration::from_secs(10))));
        let active_event = Arc::new(Mutex::new(ActiveEvent::None));
        let audio = AudioParams {
            sample_rate: 8000,
            channels: 1,
        };
        let mut pending = None;

        for _ in 0..5 {
            motion_gate.evaluate(&background_frame()).unwrap();
        }

        try_start_recording(
            &config,
            &mut motion_gate,
            &mut detector,
            &ring_buffer,
            &active_event,
            &audio,
            &background_frame(),
            Instant::now(),
            &mut pending,
            None,
        )
        .unwrap();

        assert!(!active_event.lock().unwrap().is_some());
    }

    #[test]
    #[ignore = "requires models/yolov8n.onnx + a working ONNX Runtime build; run explicitly with \
                `cargo test -- --ignored` on a machine that has both"]
    fn try_start_recording_motion_without_confirmation_does_not_start_a_recording() {
        let _guard = detect::MODEL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());
        let mut motion_gate = MotionGate::new(config.motion_threshold).unwrap();
        let mut detector = test_detector();
        let ring_buffer = Arc::new(Mutex::new(RingBuffer::new(Duration::from_secs(10))));
        let active_event = Arc::new(Mutex::new(ActiveEvent::None));
        let audio = AudioParams {
            sample_rate: 8000,
            channels: 1,
        };
        let mut pending = None;

        for _ in 0..5 {
            motion_gate.evaluate(&background_frame()).unwrap();
        }

        // A synthetic changed-region frame trips motion but has no real
        // subject, so poll_confirmed_detections returns None; no recording
        // should start regardless of how many times this polls.
        try_start_recording(
            &config,
            &mut motion_gate,
            &mut detector,
            &ring_buffer,
            &active_event,
            &audio,
            &changed_frame(),
            Instant::now(),
            &mut pending,
            None,
        )
        .unwrap();
        try_start_recording(
            &config,
            &mut motion_gate,
            &mut detector,
            &ring_buffer,
            &active_event,
            &audio,
            &changed_frame(),
            Instant::now(),
            &mut pending,
            None,
        )
        .unwrap();

        assert!(!active_event.lock().unwrap().is_some());
    }

    #[test]
    #[ignore = "requires models/yolov8n.onnx + a working ONNX Runtime build; run explicitly with \
                `cargo test -- --ignored` on a machine that has both"]
    fn try_start_recording_skips_when_ring_buffer_has_no_frames_yet() {
        // Even a confirmed detection must not start a recording if the ring
        // buffer's pre-buffer snapshot is empty (e.g. trigger fired at
        // startup before the camera produced anything); this is exercised
        // directly by constructing pending_confirmation as already-confirmable
        // and calling with an empty ring buffer, rather than relying on a
        // real confirmed detection (which a synthetic frame can't reliably
        // produce). If poll_confirmed_detections returns None here (the
        // common case for a synthetic frame), this test still validates the
        // no-op path; if it happens to return Some, the empty-buffer guard is
        // exercised for real.
        let _guard = detect::MODEL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());
        let mut motion_gate = MotionGate::new(config.motion_threshold).unwrap();
        let mut detector = test_detector();
        let ring_buffer = Arc::new(Mutex::new(RingBuffer::new(Duration::from_secs(10))));
        let active_event = Arc::new(Mutex::new(ActiveEvent::None));
        let audio = AudioParams {
            sample_rate: 8000,
            channels: 1,
        };
        let mut pending = None;

        for _ in 0..5 {
            motion_gate.evaluate(&background_frame()).unwrap();
        }

        try_start_recording(
            &config,
            &mut motion_gate,
            &mut detector,
            &ring_buffer,
            &active_event,
            &audio,
            &changed_frame(),
            Instant::now(),
            &mut pending,
            None,
        )
        .unwrap();

        assert!(!active_event.lock().unwrap().is_some());
    }

    #[test]
    #[ignore = "requires models/yolov8n.onnx + a working ONNX Runtime build; run explicitly with \
                `cargo test -- --ignored` on a machine that has both"]
    fn try_start_recording_skips_when_reconnected_too_recently() {
        // A camera reconnect just happened (reconnected_at = now), so
        // `pre_buffer_ready` is false and `try_start_recording` must skip
        // YOLO inference entirely (never calling `poll_confirmed_detections`
        // at all) rather than run it and discard the result. `Detector`
        // still needs a real loaded model to construct at
        // all (there is no stub constructor), hence this test still needs
        // `MODEL_TEST_LOCK` and is `#[ignore]`'d, but unlike before, the
        // guard being held no longer matters for whether the gate itself is
        // exercised: `detector.detect` is never invoked on this path, so the
        // assertions below hold deterministically rather than depending on
        // what a synthetic frame happens to classify as.
        let _guard = detect::MODEL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let dir = tempfile::tempdir().unwrap();
        let config = test_config(dir.path());
        let mut motion_gate = MotionGate::new(config.motion_threshold).unwrap();
        let mut detector = test_detector();
        let ring_buffer = Arc::new(Mutex::new(RingBuffer::new(Duration::from_secs(10))));
        ring_buffer.lock().unwrap().push_frame(background_frame());
        let active_event = Arc::new(Mutex::new(ActiveEvent::None));
        let audio = AudioParams {
            sample_rate: 8000,
            channels: 1,
        };
        let mut pending = Some(PendingConfirmation {
            class_name: "person",
            first_seen: Instant::now(),
        });

        for _ in 0..5 {
            motion_gate.evaluate(&background_frame()).unwrap();
        }

        try_start_recording(
            &config,
            &mut motion_gate,
            &mut detector,
            &ring_buffer,
            &active_event,
            &audio,
            &changed_frame(),
            Instant::now(),
            &mut pending,
            Some(Instant::now()),
        )
        .unwrap();

        assert!(!active_event.lock().unwrap().is_some());
        // `pending_confirmation` must survive untouched across the hold-back,
        // per the guarantee documented at the `pre_buffer_ready` call site:
        // a still-present subject should simply re-confirm and retry on the
        // next poll instead of needing a fresh two-poll confirmation cycle
        // once the buffer is ready.
        assert_eq!(pending.map(|p| p.class_name), Some("person"));
    }
}
