use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use crate::sidecar::{DetectionRecord, MotionEvent};

/// How long the camera may go without delivering a real frame before
/// `camera_stalled` reports true. Must be well above ordinary jitter between
/// detection-loop polls (camera delivery timing, YOLO inference duration can
/// both momentarily exceed one tick) so normal operation never false-trips
/// this, while still being short enough that a genuinely dead camera ends
/// the recording within a couple of seconds rather than dragging on.
///
/// Also reused by `main`'s pre-trigger staleness check so both paths agree on
/// what counts as a stalled camera.
pub const MAX_FRAME_STALL: Duration = Duration::from_millis(1500);

/// Duration of one frame-rate tick at `frame_rate` frames per second. Shared
/// by `ClipState::frame_tick` and `ffmpeg::resample_to_frame_rate` so the two
/// never compute this independently and risk diverging.
pub fn frame_tick(frame_rate: u32) -> Duration {
    Duration::from_secs_f64(1.0 / f64::from(frame_rate))
}

/// Pure timing/bookkeeping state for a single recorded clip, split out from
/// `RecordingEvent` so it can be unit-tested by constructing a bare
/// `ClipState` directly, with no ffmpeg process or temp files involved. Holds
/// everything about a clip's lifecycle *except* the actual I/O handles
/// (`RecordingEvent`'s `ffmpeg_video`/`audio_file`/temp-file paths), which
/// `RecordingEvent` still owns and drives through this struct's methods.
pub struct ClipState {
    /// Configured video frame rate for this clip.
    pub frame_rate: u32,
    /// Capture timestamp of the clip's first frame (the start of the
    /// pre-buffer window, not the trigger instant), used as the zero point
    /// for `DetectionRecord::offset_secs`.
    clip_timeline_start: Instant,
    /// When the most recent triggering detection occurred (drives the post-buffer quiet window).
    pub last_trigger_at: Instant,
    /// Timestamp up to which audio has already been drained.
    pub last_audio_drain_at: Instant,
    /// Timestamp up to which frames have already been drained.
    pub last_frame_drain_at: Instant,
    /// The next frame-rate tick due to be written, if any.
    pub next_frame_due: Option<Instant>,
    /// Wall-clock time a real, camera-delivered frame was last written. Used
    /// by `camera_stalled` to detect when the camera has stopped delivering
    /// frames. A security recording must never paper over a gap by
    /// fabricating footage, so unlike a naive resampler this never duplicates
    /// a frame to fill a missed tick; it reports the stall instead.
    pub last_real_frame_at: Instant,
    /// Every distinct class detected so far during this clip (ADR 4: the
    /// final filename must reflect every class seen over the clip's
    /// lifetime, not just the classes that triggered it).
    pub all_classes: BTreeSet<&'static str>,
    /// Every detection recorded so far during this clip.
    pub detections: Vec<DetectionRecord>,
    /// Every motion-gate trip recorded so far during this clip.
    pub motion_events: Vec<MotionEvent>,
}

impl ClipState {
    /// Starts fresh bookkeeping for a clip beginning now, with playback at `frame_rate`.
    pub fn new(frame_rate: u32, clip_timeline_start: Instant) -> Self {
        let now = Instant::now();

        Self {
            frame_rate,
            clip_timeline_start,
            last_trigger_at: now,
            last_audio_drain_at: now,
            last_frame_drain_at: now,
            next_frame_due: None,
            last_real_frame_at: now,
            all_classes: BTreeSet::new(),
            detections: Vec::new(),
            motion_events: Vec::new(),
        }
    }

    /// True once the camera has gone `MAX_FRAME_STALL` without delivering a
    /// real frame. Callers should treat this as a signal to end the
    /// recording: `drain_frames` never fabricates a frame to cover for the
    /// camera, and the motion gate has nothing new to evaluate once frames
    /// stop arriving, so nothing else will naturally close the clip.
    pub fn camera_stalled(&self) -> bool {
        self.last_real_frame_at.elapsed() >= MAX_FRAME_STALL
    }

    /// Duration of one frame-rate tick at this event's configured `frame_rate`.
    pub fn frame_tick(&self) -> Duration {
        frame_tick(self.frame_rate)
    }

    /// Decides whether a newly-drained frame at `frame_timestamp` is due to
    /// be written at this event's configured `frame_rate`, advancing
    /// `next_frame_due` to the following tick if so. The first call with no
    /// prior `next_frame_due` treats `frame_timestamp` itself as due, so
    /// draining never stalls waiting for a tick boundary that predates the
    /// first frame it ever saw.
    pub fn should_write_frame(&mut self, frame_timestamp: Instant) -> bool {
        let due = self.next_frame_due.unwrap_or(frame_timestamp);

        if frame_timestamp < due {
            return false;
        }

        #[allow(
            clippy::arithmetic_side_effects,
            reason = "Instant + a sub-second Duration only overflows after ~584 billion years of process uptime"
        )]
        let next_due = due + self.frame_tick();

        self.next_frame_due = Some(next_due);

        true
    }

    /// Anchors the frame-rate tick clock to `last_seeded_timestamp` (the
    /// capture timestamp of the last frame written by `seed`), so the first
    /// live `drain_frames` call after seeding schedules its next write one
    /// tick after the pre-buffer's last frame rather than from whatever
    /// wall-clock instant `drain_frames` first happens to run at.
    pub fn anchor_next_tick_after_seed(&mut self, last_seeded_timestamp: Instant) {
        self.last_frame_drain_at = last_seeded_timestamp;

        #[allow(
            clippy::arithmetic_side_effects,
            reason = "Instant + a sub-second Duration only overflows after ~584 billion years of process uptime"
        )]
        let next_due = last_seeded_timestamp + self.frame_tick();

        self.next_frame_due = Some(next_due);
    }

    /// Records a detection into the sidecar and resets the post-buffer quiet
    /// window. `frame_timestamp` is the capture timestamp of the frame the
    /// detection ran against (from the ring buffer), used to compute
    /// `offset_secs` relative to the clip's actual timeline start rather than
    /// wall-clock time at the moment this function happens to run.
    pub fn record_detection(
        &mut self,
        class_name: &'static str,
        confidence: f32,
        frame_timestamp: Instant,
    ) {
        let offset_secs = self.offset_secs(frame_timestamp);

        self.all_classes.insert(class_name);

        self.detections.push(DetectionRecord {
            offset_secs,
            class_name: class_name.to_string(),
            confidence,
        });

        self.last_trigger_at = Instant::now();
    }

    /// Resets the post-buffer quiet window without recording a new detection
    /// (e.g. motion continues but wasn't re-confirmed by YOLO this tick).
    pub fn touch(&mut self) {
        self.last_trigger_at = Instant::now();
    }

    /// Records a motion-gate trip into the sidecar for later audit (e.g.
    /// distinguishing a fan/curtain repeatedly tripping the gate from an
    /// actual subject). Purely diagnostic bookkeeping. It does not itself
    /// touch the post-buffer quiet window. Callers already call
    /// `touch`/`record_detection` for that as needed.
    pub fn record_motion(&mut self, changed_ratio: f32, frame_timestamp: Instant) {
        let offset_secs = self.offset_secs(frame_timestamp);

        self.motion_events.push(MotionEvent {
            offset_secs,
            changed_ratio,
        });
    }

    /// Seconds from the clip's actual timeline start (not wall-clock time at
    /// the moment the caller happens to run) to `frame_timestamp`, the
    /// capture timestamp of the frame being recorded against.
    pub fn offset_secs(&self, frame_timestamp: Instant) -> f64 {
        frame_timestamp
            .saturating_duration_since(self.clip_timeline_start)
            .as_secs_f64()
    }

    /// How long it's been since the last trigger/touch.
    pub fn quiet_for(&self) -> Duration {
        self.last_trigger_at.elapsed()
    }

    /// Consumes this state, handing ownership of the accumulated class list
    /// and sidecar records to the caller. Used by `RecordingEvent::finish`
    /// to build the final filename's class list and the `Sidecar` written
    /// alongside it.
    pub fn into_parts(
        self,
    ) -> (
        BTreeSet<&'static str>,
        Vec<DetectionRecord>,
        Vec<MotionEvent>,
    ) {
        (self.all_classes, self.detections, self.motion_events)
    }
}

/// Test-only surface, kept in a separate `impl` block (rather than mixed
/// into the block above) so production code never has a test-only method
/// sitting alongside it, per ADR 6/7's convention of splitting
/// coverage/test-only concerns into their own block/file rather than
/// interleaving them with the code real callers use.
#[cfg(test)]
impl ClipState {
    /// Backdates `last_real_frame_at` so `camera_stalled` reads true without
    /// a real `thread::sleep`, matching this file's own
    /// `camera_stalled_true_once_max_frame_stall_elapses` test below and
    /// ADR 7's convention of backdating `Instant`s instead of sleeping in
    /// real time (real sleeps at durations this close to `MAX_FRAME_STALL`
    /// are flaky under `cargo test`'s parallel load).
    #[allow(
        clippy::arithmetic_side_effects,
        clippy::unchecked_time_subtraction,
        reason = "test assertions favor unwrap/indexing/plain time arithmetic for clarity; test \
                   durations are small hardcoded constants, so underflow is not reachable"
    )]
    pub(crate) fn backdate_last_real_frame_at_past_stall(&mut self) {
        self.last_real_frame_at = Instant::now() - MAX_FRAME_STALL - Duration::from_millis(50);
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for `ClipState`'s bookkeeping (no ffmpeg process required).
    //! `Sidecar`/`DetectionRecord`/`MotionEvent` serialization is tested in
    //! `sidecar.rs`, where those types now live.
    #![allow(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        clippy::missing_panics_doc,
        clippy::unchecked_time_subtraction,
        reason = "test assertions favor unwrap/indexing/plain time arithmetic for clarity; test \
                   durations are small hardcoded constants, so underflow is not reachable"
    )]

    use super::*;

    // --- ClipState ---

    #[test]
    fn camera_stalled_false_when_recently_touched() {
        let state = ClipState::new(15, Instant::now());
        assert!(!state.camera_stalled());
    }

    #[test]
    fn camera_stalled_true_once_max_frame_stall_elapses() {
        let mut state = ClipState::new(15, Instant::now());
        state.backdate_last_real_frame_at_past_stall();
        assert!(state.camera_stalled());
    }

    #[test]
    fn quiet_for_reflects_time_since_touch() {
        let mut state = ClipState::new(15, Instant::now());
        state.last_trigger_at = Instant::now() - Duration::from_secs(5);

        assert!(state.quiet_for() >= Duration::from_secs(5));

        state.touch();
        assert!(state.quiet_for() < Duration::from_secs(1));
    }

    #[test]
    fn quiet_for_reflects_time_since_record_detection() {
        let mut state = ClipState::new(15, Instant::now());
        state.last_trigger_at = Instant::now() - Duration::from_secs(5);

        state.record_detection("person", 0.9, Instant::now());

        assert!(state.quiet_for() < Duration::from_secs(1));
    }

    #[test]
    fn offset_secs_relative_to_clip_timeline_start() {
        let start = Instant::now();
        let state = ClipState::new(15, start);

        let five_secs_in = start + Duration::from_secs(5);
        assert!((state.offset_secs(five_secs_in) - 5.0).abs() < 0.01);
    }

    #[test]
    fn offset_secs_clamps_to_zero_for_timestamp_before_clip_start() {
        let start = Instant::now();
        let state = ClipState::new(15, start);

        let before_start = start - Duration::from_secs(2);
        assert!((state.offset_secs(before_start) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn record_detection_deduplicates_and_sorts_classes() {
        let mut state = ClipState::new(15, Instant::now());
        let now = Instant::now();

        state.record_detection("dog", 0.5, now);
        state.record_detection("person", 0.9, now);
        state.record_detection("dog", 0.6, now);

        let classes: Vec<&str> = state.all_classes.iter().copied().collect();
        assert_eq!(classes, vec!["dog", "person"]);
        assert_eq!(state.detections.len(), 3);
    }

    #[test]
    fn record_motion_appends_without_touching_quiet_window() {
        let mut state = ClipState::new(15, Instant::now());
        state.last_trigger_at = Instant::now() - Duration::from_secs(10);

        state.record_motion(0.05, Instant::now());

        assert_eq!(state.motion_events.len(), 1);
        assert!(state.quiet_for() >= Duration::from_secs(9));
    }

    #[test]
    fn should_write_frame_true_on_first_call_with_no_prior_due_tick() {
        let mut state = ClipState::new(15, Instant::now());
        assert!(state.next_frame_due.is_none());

        assert!(state.should_write_frame(Instant::now()));
        assert!(state.next_frame_due.is_some());
    }

    #[test]
    fn should_write_frame_false_before_the_next_due_tick() {
        let mut state = ClipState::new(15, Instant::now());
        let first = Instant::now();
        assert!(state.should_write_frame(first));

        let just_after = first + Duration::from_millis(1);
        assert!(!state.should_write_frame(just_after));
    }

    #[test]
    fn should_write_frame_true_once_the_next_tick_is_reached() {
        let mut state = ClipState::new(15, Instant::now());
        let first = Instant::now();
        assert!(state.should_write_frame(first));

        let next_tick = first + state.frame_tick();
        assert!(state.should_write_frame(next_tick));
    }

    #[test]
    fn anchor_next_tick_after_seed_schedules_one_tick_past_the_seeded_frame() {
        let mut state = ClipState::new(15, Instant::now());
        let last_seeded = Instant::now();
        let tick = state.frame_tick();

        state.anchor_next_tick_after_seed(last_seeded);

        assert_eq!(state.last_frame_drain_at, last_seeded);
        assert_eq!(state.next_frame_due, Some(last_seeded + tick));
    }
}
