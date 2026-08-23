use std::process::{Child, Command, Stdio};

use anyhow::{Context, Result, bail};

use crate::buffer::TimestampedFrame;
use crate::clip_state;

/// Spawns ffmpeg to encode raw RGB frames fed via stdin into an H.264 file.
///
/// `Command::spawn`'s exec-failure arm is only reachable if `ffmpeg` is
/// absent from `PATH`; see `RecordingEvent::start`'s doc comment for why
/// that's not safely fakeable from a test.
pub fn spawn_video_encoder(
    path: &std::path::Path,
    width: u32,
    height: u32,
    frame_rate: u32,
) -> Result<Child> {
    #[cfg(unix)]
    use std::os::unix::process::CommandExt;

    let mut command = Command::new("ffmpeg");

    command
        .args(["-y", "-loglevel", "error"])
        .args(["-f", "rawvideo", "-pixel_format", "rgb24"])
        .args(["-video_size", &format!("{width}x{height}")])
        .args(["-framerate", &frame_rate.to_string()])
        .args(["-i", "-"])
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(path)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    // Put ffmpeg in its own process group so a terminal SIGINT (Ctrl+C)
    // doesn't reach it directly. It shares the foreground process group
    // with motioncap by default, so without this, Ctrl+C kills ffmpeg at
    // the same instant as motioncap's own ctrlc handler tries to close it
    // gracefully (closing stdin, then waiting), racing ffmpeg's own SIGINT
    // handling and producing a nonzero exit even when the output file is
    // actually complete and valid. Graceful shutdown should be the only
    // thing that ever tells ffmpeg to stop.
    #[cfg(unix)]
    command.process_group(0);

    command
        .spawn()
        .context("failed to spawn ffmpeg video encoder")
}

/// Muxes the buffered raw audio into the encoded video. `-shortest` is
/// deliberately not used: with independently-accumulated video/audio streams,
/// whichever stream is shorter due to minor drift would otherwise have the
/// *other* stream silently truncated to match, losing recorded content.
/// Instead, the audio stream is padded with silence (`apad`) to at least the
/// video's duration and `-shortest` is applied only to that padded output, so
/// the result is exactly the video's length with no dropped video frames.
///
/// `.output()`'s exec-failure arm is only reachable if `ffmpeg` is absent
/// from `PATH`; see `RecordingEvent::start`'s doc comment for why that's not
/// safely fakeable from a test.
pub fn mux_audio_into_video(
    video_path: &std::path::Path,
    audio_path: &std::path::Path,
    output_path: &std::path::Path,
    sample_rate: u32,
    channels: u16,
) -> Result<()> {
    let output = Command::new("ffmpeg")
        .args(["-y", "-i"])
        .arg(video_path)
        .args([
            "-f",
            "f32le",
            "-ar",
            &sample_rate.to_string(),
            "-ac",
            &channels.to_string(),
        ])
        .arg("-i")
        .arg(audio_path)
        .args(["-c:v", "copy", "-af", "apad", "-c:a", "aac", "-shortest"])
        .arg(output_path)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .context("failed to spawn ffmpeg for audio muxing")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);

        bail!(
            "ffmpeg audio mux exited with {}: {}",
            output.status,
            stderr.trim()
        );
    }

    Ok(())
}

/// Downsamples timestamped frames to approximately `frame_rate` frames per
/// second based on their capture timestamps, keeping the first frame at or
/// after each target tick. The ring buffer accumulates frames at the
/// camera's native capture rate (which may be higher than the encoder's
/// configured `frame_rate`), so writing every buffered frame 1:1 would
/// stretch the pre-buffer's playback duration beyond its real elapsed time.
pub fn resample_to_frame_rate(
    frames: &[TimestampedFrame],
    frame_rate: u32,
) -> Vec<&TimestampedFrame> {
    let Some(first) = frames.first() else {
        return Vec::new();
    };

    if frame_rate == 0 {
        return frames.iter().collect();
    }

    let tick = clip_state::frame_tick(frame_rate);
    let mut selected = Vec::new();
    let mut next_due = first.timestamp;

    for frame in frames {
        if frame.timestamp >= next_due {
            selected.push(frame);
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "Instant += a sub-second Duration only overflows after ~584 billion years of process uptime"
            )]
            {
                next_due += tick;
            }
        }
    }

    selected
}

#[cfg(test)]
mod tests {
    //! Unit tests for `resample_to_frame_rate`. `spawn_video_encoder` and
    //! `mux_audio_into_video` are exercised indirectly through
    //! `recorder.rs`'s `RecordingEvent` round-trip and fault-injection tests,
    //! since both require a real ffmpeg subprocess and don't have meaningful
    //! standalone behavior to unit-test beyond that.
    #![allow(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        clippy::missing_panics_doc,
        reason = "test assertions favor unwrap/indexing for clarity; panics here fail the test, which is the intended behavior"
    )]

    use std::time::{Duration, Instant};

    use super::*;

    fn frame_at(timestamp: Instant) -> TimestampedFrame {
        TimestampedFrame {
            timestamp,
            image: std::sync::Arc::new(image::RgbImage::new(1, 1)),
        }
    }

    #[test]
    fn resample_empty_input_yields_empty_output() {
        let frames: Vec<TimestampedFrame> = Vec::new();
        assert!(resample_to_frame_rate(&frames, 15).is_empty());
    }

    #[test]
    fn resample_zero_frame_rate_passes_through_every_frame() {
        let start = Instant::now();
        let frames = vec![
            frame_at(start),
            frame_at(start + Duration::from_millis(1)),
            frame_at(start + Duration::from_millis(2)),
        ];

        assert_eq!(resample_to_frame_rate(&frames, 0).len(), 3);
    }

    #[test]
    fn resample_dedups_frames_tighter_than_one_tick() {
        let start = Instant::now();
        // 15fps tick is ~66.7ms; these arrive far tighter than that.
        let frames = vec![
            frame_at(start),
            frame_at(start + Duration::from_millis(5)),
            frame_at(start + Duration::from_millis(10)),
        ];

        assert_eq!(resample_to_frame_rate(&frames, 15).len(), 1);
    }

    #[test]
    fn resample_keeps_all_frames_spaced_wider_than_one_tick() {
        let start = Instant::now();
        let frames = vec![
            frame_at(start),
            frame_at(start + Duration::from_millis(100)),
            frame_at(start + Duration::from_millis(200)),
        ];

        assert_eq!(resample_to_frame_rate(&frames, 15).len(), 3);
    }
}
