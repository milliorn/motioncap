use std::collections::VecDeque;
use std::time::{Duration, Instant};

use image::RgbImage;

#[derive(Clone)]
pub struct TimestampedFrame {
    pub timestamp: Instant,
    pub image: RgbImage,
}

#[derive(Clone)]
pub struct TimestampedAudio {
    pub timestamp: Instant,
    pub samples: Vec<f32>,
}

/// Rolling window of the last `retention` worth of frames and audio, so a
/// recording event can be started with footage from *before* the trigger fired.
pub struct RingBuffer {
    retention: Duration,
    frames: VecDeque<TimestampedFrame>,
    audio: VecDeque<TimestampedAudio>,
}

impl RingBuffer {
    pub fn new(retention: Duration) -> Self {
        Self {
            retention,
            frames: VecDeque::new(),
            audio: VecDeque::new(),
        }
    }

    pub fn push_frame(&mut self, image: RgbImage) {
        let now = Instant::now();
        self.frames.push_back(TimestampedFrame {
            timestamp: now,
            image,
        });
        self.evict_frames(now);
    }

    pub fn push_audio(&mut self, samples: Vec<f32>) {
        let now = Instant::now();
        self.audio.push_back(TimestampedAudio {
            timestamp: now,
            samples,
        });
        self.evict_audio(now);
    }

    fn evict_frames(&mut self, now: Instant) {
        while let Some(front) = self.frames.front() {
            if now.duration_since(front.timestamp) > self.retention {
                self.frames.pop_front();
            } else {
                break;
            }
        }
    }

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

    pub fn latest_frame(&self) -> Option<&TimestampedFrame> {
        self.frames.back()
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
