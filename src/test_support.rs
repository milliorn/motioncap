//! Test fixtures shared across more than one module's test suite. Kept
//! separate from any single module's `#[cfg(test)] mod tests` so neither
//! `event_lifecycle` nor `triggering` has to duplicate the other's helper or
//! reach into the other's private test module to use it.
#![cfg(test)]

use crate::buffer::TimestampedFrame;
use crate::paths::clip_path;
use crate::recorder::{RecordingEvent, RecordingEventParams};

/// Frame width/height used by every test fixture that needs a
/// `RecordingEvent`/frame but doesn't care about real dimensions: libx264
/// requires even width/height, and 2 is the smallest value that satisfies
/// that. Shared across every file's test module that constructs one of
/// these fixtures, so they can never drift into an odd (thus
/// libx264-incompatible) dimension independently.
pub const TEST_FRAME_DIM: u32 = 2;

/// Frame rate used by every `test_recording_event`/`test_pending_recording_event`
/// fixture; low enough to keep pre-buffer/tick-related test timings small.
pub const TEST_FRAME_RATE: u32 = 5;

/// Audio sample rate used by every such fixture.
pub const TEST_AUDIO_SAMPLE_RATE: u32 = 8000;

/// Audio channel count used by every such fixture (mono).
pub const TEST_AUDIO_CHANNELS: u16 = 1;

/// Starts a real (ffmpeg-backed) `RecordingEvent` (see `TEST_FRAME_DIM`) in
/// `dir`, unseeded, for tests that need to exercise the `Pending` (not yet
/// seeded) state itself, e.g. wrapping the result in `ActiveEvent::Pending`
/// to test `seed_and_drain_active_event`. Most tests want an already-seeded
/// event instead; see `test_recording_event`.
///
/// Takes `clip_timeline_start` as a parameter, rather than generating its own
/// `Instant::now()`, so a caller that goes on to seed the event (as
/// `test_recording_event` does) can timestamp that first frame at exactly
/// `clip_timeline_start` per `RecordingEventParams`'s contract ("the capture
/// timestamp of the earliest pre-buffer frame"), instead of a slightly later
/// `Instant` that would throw off `offset_secs` for that frame.
#[allow(
    clippy::unwrap_used,
    reason = "test fixture; panics here fail whichever test called it, which is the intended behavior"
)]
pub fn test_pending_recording_event(
    dir: &std::path::Path,
    clip_timeline_start: std::time::Instant,
) -> RecordingEvent {
    let started_at = chrono::Local::now();
    let path = clip_path(dir, started_at, &[]).unwrap();

    RecordingEvent::start(RecordingEventParams {
        final_clip_path: path,
        output_dir: dir.to_path_buf(),
        started_at,
        width: TEST_FRAME_DIM,
        height: TEST_FRAME_DIM,
        frame_rate: TEST_FRAME_RATE,
        audio_sample_rate: TEST_AUDIO_SAMPLE_RATE,
        audio_channels: TEST_AUDIO_CHANNELS,
        clip_timeline_start,
    })
    .unwrap()
}

/// Starts a real (ffmpeg-backed) `RecordingEvent` (see `TEST_FRAME_DIM`) in
/// `dir` and seeds it with one frame, for tests that need an actual
/// `ActiveEvent::Active` rather than `None`.
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
    let mut event = test_pending_recording_event(dir, clip_timeline_start);

    event
        .seed(
            &[TimestampedFrame {
                timestamp: clip_timeline_start,
                image: std::sync::Arc::new(image::RgbImage::new(TEST_FRAME_DIM, TEST_FRAME_DIM)),
            }],
            &[],
        )
        .unwrap();

    event
}
