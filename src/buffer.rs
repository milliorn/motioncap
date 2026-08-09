use std::collections::VecDeque;
use std::time::{Duration, Instant};

use image::RgbImage;

/// A captured video frame paired with the instant it arrived.
#[derive(Clone)]
pub struct TimestampedFrame {
    /// When this frame was captured.
    pub timestamp: Instant,
    /// The captured frame's decoded pixel data.
    pub image: RgbImage,
}

/// A chunk of captured audio samples paired with the instant it arrived.
#[derive(Clone)]
pub struct TimestampedAudio {
    /// When this audio chunk was captured.
    pub timestamp: Instant,
    /// Interleaved PCM samples for this chunk.
    pub samples: Vec<f32>,
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
        while let Some(front) = self.frames.front() {
            if now.duration_since(front.timestamp) > self.retention {
                self.frames.pop_front();
            } else {
                break;
            }
        }
    }

    /// Drops audio chunks older than `retention` relative to `now`.
    fn evict_audio(&mut self, now: Instant) {
        while let Some(front) = self.audio.front() {
            if now.duration_since(front.timestamp) > self.retention {
                self.audio.pop_front();
            } else {
                break;
            }
        }
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
    /// polls instead of only ever reading `latest_frame` -- reading just the
    /// latest silently skips any frame that arrived and was superseded
    /// before the writer's next poll, which shows up as a visible jump in
    /// the subject's position despite otherwise-correct frame timestamps.
    pub fn frames_since(&self, since: Instant) -> Vec<TimestampedFrame> {
        self.frames
            .iter()
            .filter(|frame| frame.timestamp > since)
            .cloned()
            .collect()
    }

    /// Audio chunks pushed strictly after `since`, oldest first. Used to drain
    /// newly-captured audio into an active recording each poll, so live clips
    /// keep accumulating audio for their full duration instead of only ever
    /// containing the pre-buffer's audio.
    pub fn audio_since(&self, since: Instant) -> Vec<TimestampedAudio> {
        self.audio
            .iter()
            .filter(|chunk| chunk.timestamp > since)
            .cloned()
            .collect()
    }
}
