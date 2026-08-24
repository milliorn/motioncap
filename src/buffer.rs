use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use image::RgbImage;

/// Common shape shared by every kind of ring-buffer entry (frames, audio),
/// so eviction and since-filtering can be written once and reused for both
/// rather than duplicated per entry type.
trait Timestamped {
    /// When this entry was captured.
    fn timestamp(&self) -> Instant;
}

/// A captured video frame paired with the instant it arrived.
///
/// `image` is `Arc`-wrapped so that draining/filtering a batch of frames out
/// of the ring buffer (`frames_since`, used by the 15fps writer loop; also
/// `latest_frame().cloned()` on the detection/preview polling paths) clones a
/// refcount instead of a full decoded pixel buffer. Measured directly: at
/// 1920x1080 with a 5-frame writer backlog, cloning the owned `RgbImage`
/// costs ~6.9ms per poll (over 10% of one 66.7ms 15fps tick budget), scaling
/// to ~14.9ms (22%) at 3840x2160; the `Arc` clone costs under 150ns
/// regardless of resolution or backlog depth in the same benchmark.
#[derive(Clone)]
pub struct TimestampedFrame {
    /// When this frame was captured.
    pub timestamp: Instant,
    /// The captured frame's decoded pixel data.
    pub image: Arc<RgbImage>,
}

impl Timestamped for TimestampedFrame {
    fn timestamp(&self) -> Instant {
        self.timestamp
    }
}

/// A chunk of captured audio samples paired with the instant it arrived.
#[derive(Clone)]
pub struct TimestampedAudio {
    /// When this audio chunk was captured.
    pub timestamp: Instant,
    /// Interleaved PCM samples for this chunk.
    pub samples: Vec<f32>,
}

impl Timestamped for TimestampedAudio {
    fn timestamp(&self) -> Instant {
        self.timestamp
    }
}

/// Drops entries older than `retention` relative to `now`, oldest first.
/// Shared by `RingBuffer::evict_frames`/`evict_audio` so the two never
/// implement the eviction rule independently.
fn evict<T: Timestamped>(deque: &mut VecDeque<T>, now: Instant, retention: Duration) {
    while let Some(front) = deque.front() {
        if now.duration_since(front.timestamp()) > retention {
            deque.pop_front();
        } else {
            break;
        }
    }
}

/// Entries pushed strictly after `since`, oldest first. Shared by
/// `RingBuffer::frames_since`/`audio_since` so the two never implement the
/// same filter independently.
fn since<T: Timestamped + Clone>(deque: &VecDeque<T>, since: Instant) -> Vec<T> {
    deque
        .iter()
        .filter(|entry| entry.timestamp() > since)
        .cloned()
        .collect()
}

/// Rolling window of the last `retention` worth of frames and audio, so a
/// recording event can be started with footage from *before* the trigger fired.
pub struct RingBuffer {
    /// How long a frame/audio chunk is kept before being evicted.
    retention: Duration,
    /// Buffered video frames, oldest first.
    frames: VecDeque<TimestampedFrame>,
    /// Buffered audio chunks, oldest first.
    audio: VecDeque<TimestampedAudio>,
}

impl RingBuffer {
    /// Creates an empty buffer that retains up to `retention` worth of frames/audio.
    pub const fn new(retention: Duration) -> Self {
        Self {
            retention,
            frames: VecDeque::new(),
            audio: VecDeque::new(),
        }
    }

    /// Appends a newly-captured frame, timestamped now, and evicts stale
    /// frames. Wraps `image` in an `Arc` once here, at the single point a
    /// freshly-decoded frame enters the buffer, so every later read
    /// (`latest_frame`, `frames_since`, `snapshot`) clones a refcount instead
    /// of the pixel buffer itself.
    pub fn push_frame(&mut self, image: RgbImage) {
        let now = Instant::now();
        self.frames.push_back(TimestampedFrame {
            timestamp: now,
            image: Arc::new(image),
        });
        self.evict_frames(now);
    }

    /// Appends a newly-captured audio chunk, timestamped now, and evicts stale audio.
    pub fn push_audio(&mut self, samples: Vec<f32>) {
        let now = Instant::now();
        self.audio.push_back(TimestampedAudio {
            timestamp: now,
            samples,
        });
        self.evict_audio(now);
    }

    /// Drops frames older than `retention` relative to `now`.
    fn evict_frames(&mut self, now: Instant) {
        evict(&mut self.frames, now, self.retention);
    }

    /// Drops audio chunks older than `retention` relative to `now`.
    fn evict_audio(&mut self, now: Instant) {
        evict(&mut self.audio, now, self.retention);
    }

    /// Snapshot of everything currently buffered, oldest first. Used to seed a
    /// new recording with the pre-event window when a trigger fires.
    pub fn snapshot(&self) -> (Vec<TimestampedFrame>, Vec<TimestampedAudio>) {
        (
            self.frames.iter().cloned().collect(),
            self.audio.iter().cloned().collect(),
        )
    }

    /// The most recently pushed frame, if any.
    pub fn latest_frame(&self) -> Option<&TimestampedFrame> {
        self.frames.back()
    }

    /// Frames pushed strictly after `since`, oldest first. Used by the
    /// recording writer to drain every frame the camera produced between
    /// polls instead of only ever reading `latest_frame`: reading just the
    /// latest silently skips any frame that arrived and was superseded
    /// before the writer's next poll, which shows up as a visible jump in
    /// the subject's position despite otherwise-correct frame timestamps.
    pub fn frames_since(&self, since_ts: Instant) -> Vec<TimestampedFrame> {
        since(&self.frames, since_ts)
    }

    /// Audio chunks pushed strictly after `since`, oldest first. Used to drain
    /// newly-captured audio into an active recording each poll, so live clips
    /// keep accumulating audio for their full duration instead of only ever
    /// containing the pre-buffer's audio.
    pub fn audio_since(&self, since_ts: Instant) -> Vec<TimestampedAudio> {
        since(&self.audio, since_ts)
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for `RingBuffer`'s retention, eviction, and query behavior.
    #![allow(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        clippy::missing_panics_doc,
        reason = "test assertions favor unwrap/indexing for clarity; panics here fail the test, which is the intended behavior"
    )]

    use std::thread::sleep;

    use image::RgbImage;

    use super::*;

    /// A frame small enough that its content is irrelevant to these tests;
    /// only its timestamp and buffer membership are ever checked.
    const BLANK_FRAME_DIM: u32 = 1;

    /// Ring-buffer retention window used by tests that only care about
    /// ordering/filtering, not eviction (long enough that nothing evicts
    /// during a fast-running test).
    const AMPLE_RETENTION: Duration = Duration::from_secs(10);

    /// Ring-buffer retention window used by the eviction tests: short enough
    /// that `EVICTION_SLEEP` reliably exceeds it.
    const SHORT_RETENTION: Duration = Duration::from_millis(10);

    /// Sleep between two pushes, long enough to guarantee a distinct,
    /// strictly-later timestamp on typical `Instant` clock resolution
    /// without being so long it slows the test suite down.
    const DISTINCT_TIMESTAMP_SLEEP: Duration = Duration::from_millis(5);

    /// Sleep between two pushes in the eviction tests, long enough to
    /// reliably exceed `SHORT_RETENTION`.
    const EVICTION_SLEEP: Duration = Duration::from_millis(30);

    /// First of two distinct audio sample values, used only to distinguish
    /// "the earlier chunk" from "the later chunk" in assertions.
    const FIRST_AUDIO_SAMPLE: f32 = 0.0;

    /// Second of two distinct audio sample values, used only to distinguish
    /// "the earlier chunk" from "the later chunk" in assertions.
    const SECOND_AUDIO_SAMPLE: f32 = 1.0;

    fn blank_frame() -> RgbImage {
        RgbImage::new(BLANK_FRAME_DIM, BLANK_FRAME_DIM)
    }

    #[test]
    fn new_buffer_has_no_latest_frame() {
        let buffer = RingBuffer::new(AMPLE_RETENTION);
        assert!(buffer.latest_frame().is_none());
    }

    #[test]
    fn latest_frame_returns_most_recently_pushed() {
        let mut buffer = RingBuffer::new(AMPLE_RETENTION);
        buffer.push_frame(blank_frame());
        buffer.push_frame(blank_frame());
        let latest = buffer.latest_frame();
        assert!(latest.is_some());
    }

    #[test]
    fn snapshot_returns_oldest_first() {
        let mut buffer = RingBuffer::new(AMPLE_RETENTION);
        buffer.push_frame(blank_frame());
        sleep(DISTINCT_TIMESTAMP_SLEEP);
        buffer.push_frame(blank_frame());

        let (frames, _) = buffer.snapshot();
        assert_eq!(frames.len(), 2);
        assert!(frames[0].timestamp <= frames[1].timestamp);
    }

    #[test]
    fn frames_since_excludes_frame_exactly_at_since() {
        let mut buffer = RingBuffer::new(AMPLE_RETENTION);
        buffer.push_frame(blank_frame());
        let since = buffer.latest_frame().unwrap().timestamp;

        assert!(buffer.frames_since(since).is_empty());
    }

    #[test]
    fn frames_since_includes_frames_strictly_after() {
        let mut buffer = RingBuffer::new(AMPLE_RETENTION);
        buffer.push_frame(blank_frame());
        let since = buffer.latest_frame().unwrap().timestamp;
        sleep(DISTINCT_TIMESTAMP_SLEEP);
        buffer.push_frame(blank_frame());

        assert_eq!(buffer.frames_since(since).len(), 1);
    }

    #[test]
    fn audio_since_excludes_chunk_exactly_at_since() {
        let mut buffer = RingBuffer::new(AMPLE_RETENTION);
        buffer.push_audio(vec![FIRST_AUDIO_SAMPLE]);
        let since = buffer.snapshot().1.last().unwrap().timestamp;

        assert!(buffer.audio_since(since).is_empty());
    }

    #[test]
    fn audio_since_includes_chunks_strictly_after() {
        let mut buffer = RingBuffer::new(AMPLE_RETENTION);
        buffer.push_audio(vec![FIRST_AUDIO_SAMPLE]);
        let since = buffer.snapshot().1.last().unwrap().timestamp;
        sleep(DISTINCT_TIMESTAMP_SLEEP);
        buffer.push_audio(vec![SECOND_AUDIO_SAMPLE]);

        assert_eq!(buffer.audio_since(since).len(), 1);
    }

    #[test]
    fn frames_older_than_retention_are_evicted_on_next_push() {
        let mut buffer = RingBuffer::new(SHORT_RETENTION);
        buffer.push_frame(blank_frame());
        sleep(EVICTION_SLEEP);
        buffer.push_frame(blank_frame());

        let (frames, _) = buffer.snapshot();
        assert_eq!(frames.len(), 1);
    }

    #[test]
    fn audio_older_than_retention_is_evicted_on_next_push() {
        let mut buffer = RingBuffer::new(SHORT_RETENTION);
        buffer.push_audio(vec![FIRST_AUDIO_SAMPLE]);
        sleep(EVICTION_SLEEP);
        buffer.push_audio(vec![SECOND_AUDIO_SAMPLE]);

        let (_, audio) = buffer.snapshot();
        assert_eq!(audio.len(), 1);
    }
}
