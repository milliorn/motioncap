use std::collections::VecDeque;
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
#[derive(Clone)]
pub struct TimestampedFrame {
    /// When this frame was captured.
    pub timestamp: Instant,
    /// The captured frame's decoded pixel data.
    pub image: RgbImage,
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

    /// Appends a newly-captured frame, timestamped now, and evicts stale frames.
    pub fn push_frame(&mut self, image: RgbImage) {
        let now = Instant::now();
        self.frames.push_back(TimestampedFrame {
            timestamp: now,
            image,
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

    fn blank_frame() -> RgbImage {
        RgbImage::new(1, 1)
    }

    #[test]
    fn new_buffer_has_no_latest_frame() {
        let buffer = RingBuffer::new(Duration::from_secs(10));
        assert!(buffer.latest_frame().is_none());
    }

    #[test]
    fn latest_frame_returns_most_recently_pushed() {
        let mut buffer = RingBuffer::new(Duration::from_secs(10));
        buffer.push_frame(blank_frame());
        buffer.push_frame(blank_frame());
        let latest = buffer.latest_frame();
        assert!(latest.is_some());
    }

    #[test]
    fn snapshot_returns_oldest_first() {
        let mut buffer = RingBuffer::new(Duration::from_secs(10));
        buffer.push_frame(blank_frame());
        sleep(Duration::from_millis(5));
        buffer.push_frame(blank_frame());

        let (frames, _) = buffer.snapshot();
        assert_eq!(frames.len(), 2);
        assert!(frames[0].timestamp <= frames[1].timestamp);
    }

    #[test]
    fn frames_since_excludes_frame_exactly_at_since() {
        let mut buffer = RingBuffer::new(Duration::from_secs(10));
        buffer.push_frame(blank_frame());
        let since = buffer.latest_frame().unwrap().timestamp;

        assert!(buffer.frames_since(since).is_empty());
    }

    #[test]
    fn frames_since_includes_frames_strictly_after() {
        let mut buffer = RingBuffer::new(Duration::from_secs(10));
        buffer.push_frame(blank_frame());
        let since = buffer.latest_frame().unwrap().timestamp;
        sleep(Duration::from_millis(5));
        buffer.push_frame(blank_frame());

        assert_eq!(buffer.frames_since(since).len(), 1);
    }

    #[test]
    fn audio_since_excludes_chunk_exactly_at_since() {
        let mut buffer = RingBuffer::new(Duration::from_secs(10));
        buffer.push_audio(vec![0.0]);
        let since = buffer.snapshot().1.last().unwrap().timestamp;

        assert!(buffer.audio_since(since).is_empty());
    }

    #[test]
    fn audio_since_includes_chunks_strictly_after() {
        let mut buffer = RingBuffer::new(Duration::from_secs(10));
        buffer.push_audio(vec![0.0]);
        let since = buffer.snapshot().1.last().unwrap().timestamp;
        sleep(Duration::from_millis(5));
        buffer.push_audio(vec![1.0]);

        assert_eq!(buffer.audio_since(since).len(), 1);
    }

    #[test]
    fn frames_older_than_retention_are_evicted_on_next_push() {
        let mut buffer = RingBuffer::new(Duration::from_millis(10));
        buffer.push_frame(blank_frame());
        sleep(Duration::from_millis(30));
        buffer.push_frame(blank_frame());

        let (frames, _) = buffer.snapshot();
        assert_eq!(frames.len(), 1);
    }

    #[test]
    fn audio_older_than_retention_is_evicted_on_next_push() {
        let mut buffer = RingBuffer::new(Duration::from_millis(10));
        buffer.push_audio(vec![0.0]);
        sleep(Duration::from_millis(30));
        buffer.push_audio(vec![1.0]);

        let (_, audio) = buffer.snapshot();
        assert_eq!(audio.len(), 1);
    }
}
