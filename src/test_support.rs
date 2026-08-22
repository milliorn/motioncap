//! Test fixtures shared across more than one module's test suite. Kept
//! separate from any single module's `#[cfg(test)] mod tests` so neither
//! `event_lifecycle` nor `triggering` has to duplicate the other's helper or
//! reach into the other's private test module to use it.
#![cfg(test)]

use crate::buffer::TimestampedFrame;
use crate::paths::clip_path;
use crate::recorder::{RecordingEvent, RecordingEventParams};

/// Starts a real (ffmpeg-backed) 2x2 `RecordingEvent` in `dir` and seeds it
/// with one frame, for tests that need an actual `ActiveEvent::Active`
/// rather than `None`.
///
/// Seeding at least one frame is required, not cosmetic: an event
/// `finish()`ed with zero video frames written produces a `Duration: N/A`
/// video stream, and `mux_audio_into_video`'s `apad` filter has no duration
/// to pad audio *to* in that case; `-shortest` never trips, so ffmpeg pads
/// forever and the mux process hangs indefinitely (confirmed directly:
/// reproduced standalone with a zero-frame video and empty audio file,
/// `apad` ran for 8+ seconds generating unbounded output before being
/// killed). A real recording always has pre-buffer frames seeded before
/// anything can call `finish()`, so this matches production usage, not just
/// what makes the test terminate.
#[allow(
    clippy::unwrap_used,
    reason = "test fixture; panics here fail whichever test called it, which is the intended behavior"
)]
pub fn test_recording_event(dir: &std::path::Path) -> RecordingEvent {
    let clip_timeline_start = std::time::Instant::now();
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
