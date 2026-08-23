use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use anyhow::Result;

use crate::buffer::{RingBuffer, TimestampedAudio, TimestampedFrame};
use crate::recorder::RecordingEvent;
use crate::writer_drained::WriterDrained;

/// An event that's been started (ffmpeg spawned) but whose pre-event buffer
/// hasn't been written yet. Kept separate from `RecordingEvent` construction
/// (see `RecordingEvent::start`'s docs) so the detection loop never blocks on
/// writing dozens of pre-buffer frames. The writer thread seeds it as the
/// first thing it does once it sees a pending event, then it becomes a
/// normal actively-written event.
pub struct PendingEvent {
    /// The started recording (ffmpeg already spawned) awaiting its pre-buffer seed.
    pub event: RecordingEvent,
    /// Pre-trigger frames to seed into `event` once the writer thread picks it up.
    pub pre_frames: Vec<TimestampedFrame>,
    /// Pre-trigger audio to seed into `event` once the writer thread picks it up.
    pub pre_audio: Vec<TimestampedAudio>,
}

/// Shared state for the currently in-progress recording, if any. Starts as
/// `Pending` (ffmpeg spawned, pre-buffer not yet written) so the writer
/// thread can seed it without blocking whichever thread created it; once
/// seeded it becomes `Active` and receives normal steady-paced writes.
pub enum ActiveEvent {
    /// No recording is in progress.
    None,
    /// ffmpeg has been spawned but the pre-event buffer hasn't been seeded yet.
    Pending(PendingEvent),
    /// The event is seeded and receiving normal steady-paced writes.
    Active(RecordingEvent),
}

impl ActiveEvent {
    /// Whether any recording (pending or active) is currently in progress.
    pub const fn is_some(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// The active `RecordingEvent`, if one is seeded and receiving writes.
    pub const fn as_recording_mut(&mut self) -> Option<&mut RecordingEvent> {
        match self {
            Self::None | Self::Pending(_) => None,
            Self::Active(event) => Some(event),
        }
    }

    /// Takes the in-progress recording (if any), leaving `None` behind.
    pub fn take(&mut self) -> Option<RecordingEvent> {
        match std::mem::replace(self, Self::None) {
            Self::None => None,
            Self::Pending(pending) => Some(pending.event),
            Self::Active(event) => Some(event),
        }
    }
}

/// Seeds `pending`'s pre-buffer, logging (not propagating) any error, and
/// returns the now-seeded `RecordingEvent`. Shared by `finish_event_on_shutdown`
/// and `seed_and_drain_active_event`, both of which must seed a `Pending`
/// event's pre-buffer before treating it as a normal actively-written event,
/// but differ in what they do with the result afterward (one hands it
/// straight to `finish`, the other stores it back as `ActiveEvent::Active`).
fn seed_pending(mut pending: PendingEvent) -> RecordingEvent {
    if let Err(err) = pending.event.seed(&pending.pre_frames, &pending.pre_audio) {
        log::error!("failed to seed pre-buffer into new recording: {err:?}");
    }
    pending.event
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
pub fn finish_event_on_shutdown(
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
        ActiveEvent::Pending(pending) => Some(seed_pending(pending)),
        ActiveEvent::Active(event) => Some(event),
    };

    if let Some(event) = event {
        event.finish()?;
        log::info!("recording closed on shutdown");
    }

    Ok(())
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
pub fn seed_and_drain_active_event(
    ring_buffer: &Mutex<RingBuffer>,
    active_event: &Mutex<ActiveEvent>,
) {
    let mut guard = active_event.lock().expect("active event lock poisoned");

    let taken = std::mem::replace(&mut *guard, ActiveEvent::None);

    match taken {
        ActiveEvent::Pending(pending) => *guard = ActiveEvent::Active(seed_pending(pending)),
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

/// Closes the active recording if either close condition is met: the camera
/// has stalled (see `RecordingEvent::camera_stalled`), which the motion gate
/// can never detect on its own since a stalled camera means no new frames
/// ever reach it to evaluate; or the post-buffer quiet window has elapsed
/// with no fresh trigger. No-ops if neither condition holds.
///
/// Takes the `MutexGuard` by value so it can be dropped before `finish`
/// (which waits on ffmpeg) runs. The lock must not be held across that.
pub fn close_event_if_done(
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

#[cfg(test)]
mod tests {
    //! Unit tests for the `ActiveEvent` state machine and its
    //! shutdown/close/seed-drain lifecycle operations. The per-poll
    //! start/extend decision logic that drives this state machine (
    //! `evaluate_active_event`, `try_start_recording`) lives in `triggering`
    //! and has its own tests there.
    #![allow(
        clippy::unwrap_used,
        clippy::missing_panics_doc,
        reason = "test assertions favor unwrap for clarity; panics here fail the test, which is the intended behavior"
    )]

    use crate::test_support::{test_pending_recording_event, test_recording_event};

    use super::*;

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
            .state_mut()
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
        let expected_path = crate::paths::clip_path(dir.path(), started_at, &[]).unwrap();

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
        let expected_path = crate::paths::clip_path(dir.path(), started_at, &[]).unwrap();

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
        let clip_timeline_start = std::time::Instant::now();
        let event = test_pending_recording_event(dir.path(), clip_timeline_start);

        let active_event = Mutex::new(ActiveEvent::Pending(PendingEvent {
            event,
            pre_frames: vec![TimestampedFrame {
                timestamp: clip_timeline_start,
                image: std::sync::Arc::new(image::RgbImage::new(2, 2)),
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
}
