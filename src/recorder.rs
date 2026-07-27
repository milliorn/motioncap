use std::collections::BTreeSet;
use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Local};
use serde::Serialize;

use crate::buffer::{TimestampedAudio, TimestampedFrame};
use crate::paths::{clip_path, sidecar_path};

/// One recorded detection, written into a clip's `.json` sidecar (ADR 4).
#[derive(Serialize)]
pub struct DetectionRecord {
    /// Seconds from the start of the clip when this detection occurred.
    pub offset_secs: f64,
    /// The detected COCO class name.
    pub class_name: String,
    /// The model's reported confidence for this detection.
    pub confidence: f32,
}

/// A clip's `.json` sidecar contents (ADR 4).
#[derive(Serialize)]
pub struct Sidecar {
    /// Every detection recorded during the clip, in chronological order.
    pub detections: Vec<DetectionRecord>,
}

/// Construction parameters for `RecordingEvent::start`, grouped into a
/// struct since the individual values (video dimensions, audio format,
/// path-naming inputs, timeline anchor) don't share a natural owner.
pub struct RecordingEventParams {
    /// Where the finished, muxed clip is written, before any class-list
    /// rename (see `RecordingEvent::all_classes`).
    pub final_clip_path: std::path::PathBuf,
    /// Directory recordings are written under (ADR 4), needed alongside
    /// `started_at` to recompute the filename at `finish` if the class list
    /// grows over the clip's lifetime.
    pub output_dir: std::path::PathBuf,
    /// Wall-clock time the clip started; must match what `final_clip_path`
    /// was built from via `clip_path`.
    pub started_at: DateTime<Local>,
    /// Video frame width in pixels.
    pub width: u32,
    /// Video frame height in pixels.
    pub height: u32,
    /// Configured video frame rate the encoder is spawned with.
    pub frame_rate: u32,
    /// Sample rate of the audio that will be written, needed to mux correctly.
    pub audio_sample_rate: u32,
    /// Channel count of the audio that will be written, needed to mux correctly.
    pub audio_channels: u16,
    /// Capture timestamp of the clip's first frame (the start of the
    /// pre-buffer window, not the trigger instant), used as the zero point
    /// for `DetectionRecord::offset_secs`.
    pub clip_timeline_start: Instant,
}

/// Manages the lifecycle of a single recorded clip: seeds the file with the
/// pre-event buffer, accepts live frames as they arrive, and tracks the
/// post-event quiet window so the caller knows when to close it (ADR 2, ADR 4).
///
/// Audio is buffered to a temporary raw PCM file and muxed in by ffmpeg only
/// when the clip closes (`finish`), since ffmpeg needs the final duration of
/// both streams up front to mux them correctly; video frames, by contrast,
/// are streamed live via ffmpeg's stdin as they arrive.
pub struct RecordingEvent {
    /// The running ffmpeg process encoding video, fed live via its stdin.
    ffmpeg_video: Child,
    /// Path to the temporary raw PCM file audio is buffered into.
    audio_tmp_path: std::path::PathBuf,
    /// Open handle to `audio_tmp_path`, written to as audio arrives.
    audio_file: std::fs::File,
    /// Where the finished, muxed clip is written on `finish`, before any
    /// class-list rename (see `all_classes`).
    final_clip_path: std::path::PathBuf,
    /// Directory recordings are written under, needed to recompute
    /// `final_clip_path` at `finish` if `all_classes` grew since `start`.
    output_dir: std::path::PathBuf,
    /// Wall-clock time the clip started, needed (alongside `output_dir`) to
    /// recompute `final_clip_path` at `finish` via the same `clip_path`
    /// naming convention it was originally built with.
    started_at: DateTime<Local>,
    /// Path to the temporary video-only file ffmpeg encodes into.
    video_tmp_path: std::path::PathBuf,
    /// Sample rate of the buffered audio, needed to mux correctly.
    audio_sample_rate: u32,
    /// Channel count of the buffered audio, needed to mux correctly.
    audio_channels: u16,
    /// Configured video frame rate for this clip.
    frame_rate: u32,
    /// Capture timestamp of the clip's first frame (the start of the
    /// pre-buffer window, not the trigger instant), used as the zero point
    /// for `DetectionRecord::offset_secs`.
    clip_timeline_start: Instant,
    /// When the most recent triggering detection occurred (drives the post-buffer quiet window).
    last_trigger_at: Instant,
    /// Timestamp up to which audio has already been drained.
    last_audio_drain_at: Instant,
    /// Timestamp up to which frames have already been drained.
    last_frame_drain_at: Instant,
    /// The next frame-rate tick due to be written, if any.
    next_frame_due: Option<Instant>,
    /// The most recently written video frame, kept so `drain_frames` can
    /// duplicate it to fill ticks the camera didn't deliver a new frame for
    /// (see `drain_frames`'s docs).
    last_written_frame: Option<image::RgbImage>,
    /// Every distinct class detected so far during this clip (ADR 4: the
    /// final filename must reflect every class seen over the clip's
    /// lifetime, not just the classes that triggered it).
    all_classes: BTreeSet<&'static str>,
    /// Every detection recorded so far during this clip.
    detections: Vec<DetectionRecord>,
}

impl RecordingEvent {
    /// Starts a new event: launches the video-encoding ffmpeg process and
    /// creates the temp audio file. `final_clip_path` should already reflect
    /// the trigger's classes/timestamp per the output layout convention
    /// (ADR 4); this module only performs the actual encoding/muxing, not
    /// path naming.
    ///
    /// `audio_sample_rate`/`audio_channels` must match whatever format the
    /// caller's audio samples were captured/converted to (see
    /// `capture::audio::AudioStreamInfo`), since the mux step needs the real
    /// values to interpret the buffered PCM correctly.
    ///
    /// Deliberately does *not* write the pre-buffer here -- see `seed`.
    /// Writing dozens of frames synchronously on the caller's thread would
    /// block it for long enough that real wall-clock time passes with
    /// nothing being recorded, which shows up as a skip/jump right at the
    /// start of the clip once live writing resumes. Callers should invoke
    /// `seed` from whatever thread is responsible for steady-paced frame
    /// writing instead.
    ///
    /// `clip_timeline_start` should be the capture timestamp of the earliest
    /// pre-buffer frame that will be seeded into this clip (i.e. the actual
    /// start of the video file), not the trigger instant -- it's the zero
    /// point `DetectionRecord::offset_secs` is measured from, so detections
    /// recorded against footage that predates the trigger (the pre-buffer
    /// window) get correct nonzero offsets instead of all reading ~0.
    ///
    /// `output_dir`/`started_at` must be the same values `final_clip_path`
    /// was originally built from via `clip_path`, so `finish` can recompute
    /// the filename with the full class list accumulated over the clip's
    /// lifetime (ADR 4) and rename to it if it grew since `start`.
    pub fn start(params: RecordingEventParams) -> Result<Self> {
        let RecordingEventParams {
            final_clip_path,
            output_dir,
            started_at,
            width,
            height,
            frame_rate,
            audio_sample_rate,
            audio_channels,
            clip_timeline_start,
        } = params;

        let video_tmp_path = final_clip_path.with_extension("video.tmp.mp4");
        let audio_tmp_path = final_clip_path.with_extension("audio.tmp.pcm");

        let ffmpeg_video = spawn_video_encoder(&video_tmp_path, width, height, frame_rate)?;
        let audio_file = std::fs::File::create(&audio_tmp_path).with_context(|| {
            format!(
                "failed to create temp audio file {}",
                audio_tmp_path.display()
            )
        })?;

        let now = Instant::now();

        Ok(Self {
            ffmpeg_video,
            audio_tmp_path,
            audio_file,
            final_clip_path,
            output_dir,
            started_at,
            video_tmp_path,
            audio_sample_rate,
            audio_channels,
            frame_rate,
            clip_timeline_start,
            last_trigger_at: now,
            last_audio_drain_at: now,
            last_frame_drain_at: now,
            next_frame_due: None,
            last_written_frame: None,
            all_classes: BTreeSet::new(),
            detections: Vec::new(),
        })
    }

    /// Writes the pre-event buffer (frames captured before the trigger
    /// fired) into a freshly-started event. Must be called once, as the
    /// first write against a new event, before any live `write_frame`/
    /// `drain_audio` calls -- see the note on `start` for why this is
    /// separate from event construction.
    ///
    /// The ring buffer accumulates frames at the camera's native capture
    /// rate, which may be higher than this event's configured `frame_rate`
    /// (set in `start`, the rate the video encoder was spawned with).
    /// Pre-buffered frames are resampled down to that rate using their
    /// timestamps before writing, so the pre-buffer portion of the clip plays
    /// back at the correct real-time duration instead of being stretched by
    /// writing every captured frame 1:1.
    pub fn seed(
        &mut self,
        pre_frames: &[TimestampedFrame],
        pre_audio: &[TimestampedAudio],
    ) -> Result<()> {
        let selected = resample_to_frame_rate(pre_frames, self.frame_rate);

        for frame in &selected {
            self.write_frame(&frame.image)?;
            self.last_written_frame = Some(frame.image.clone());
        }

        if let Some(last) = selected.last() {
            let tick = Duration::from_secs_f64(1.0 / f64::from(self.frame_rate));

            self.last_frame_drain_at = last.timestamp;
            
            #[allow(
                clippy::arithmetic_side_effects,
                reason = "Instant + a sub-second Duration only overflows after ~584 billion years of process uptime"
            )]
            let next_due = last.timestamp + tick;

            self.next_frame_due = Some(next_due);
        }

        for chunk in pre_audio {
            self.write_audio(&chunk.samples)?;
        }

        if let Some(last) = pre_audio.last() {
            self.last_audio_drain_at = last.timestamp;
        }

        Ok(())
    }

    /// Writes every frame captured since the last drain (or since `seed`,
    /// for the first call), resampled to this event's configured frame rate
    /// using a persistent tick anchor carried across calls. Must be polled
    /// periodically while the event is active.
    ///
    /// Draining *all* new frames (not just the newest) matters because the
    /// camera's delivery rate can momentarily exceed the writer's poll rate;
    /// reading only the latest frame each poll silently skips over whatever
    /// arrived and was superseded in between, which produces a visible jump
    /// in the subject's position despite otherwise-correct frame timing.
    ///
    /// If the camera momentarily delivers *fewer* frames than `frame_rate`
    /// (or stalls), the reverse problem applies: no buffered frame may cross
    /// a given tick, so nothing gets written for it. Left unhandled, the
    /// encoded video (ffmpeg's `-framerate` assumes uniform spacing) falls
    /// behind wall-clock elapsed time, which plays back sped-up and leaves
    /// the independently wall-clock-accumulated audio longer than the video
    /// at mux time. To keep output duration aligned with real elapsed time,
    /// any tick that's fully elapsed by wall-clock `Instant::now()` without a
    /// new frame satisfying it gets filled by re-writing the last frame
    /// actually written.
    pub fn drain_frames(
        &mut self,
        ring_buffer: &std::sync::Mutex<crate::buffer::RingBuffer>,
    ) -> Result<()> {
        let new_frames = {
            let buf = ring_buffer.lock().expect("ring buffer lock poisoned");
            buf.frames_since(self.last_frame_drain_at)
        };

        if let Some(last) = new_frames.last() {
            self.last_frame_drain_at = last.timestamp;
        }

        let tick = Duration::from_secs_f64(1.0 / f64::from(self.frame_rate));
        let mut next_due = self.next_frame_due;

        for frame in &new_frames {
            let due = next_due.unwrap_or(frame.timestamp);

            if frame.timestamp >= due {
                self.write_frame(&frame.image)?;
                self.last_written_frame = Some(frame.image.clone());
                #[allow(
                    clippy::arithmetic_side_effects,
                    reason = "Instant + a sub-second Duration only overflows after ~584 billion years of process uptime"
                )]
                let new_due = due + tick;
                next_due = Some(new_due);
            }
        }

        // Fill any ticks that have fully elapsed by wall-clock time but
        // weren't satisfied above (the camera didn't deliver a qualifying
        // frame this poll), duplicating the last frame actually written so
        // encoded duration keeps pace with real elapsed time. The `now` bound
        // caps this to one tick per elapsed stall duration, not a fixed
        // count -- a camera stall of several seconds still produces a burst
        // of that many synchronous writes here before the next poll.
        let now = Instant::now();

        while let Some(due) = next_due {
            if due > now {
                break;
            }

            let Some(last_frame) = self.last_written_frame.clone() else {
                break;
            };

            self.write_frame(&last_frame)?;

            #[allow(
                clippy::arithmetic_side_effects,
                reason = "Instant + a sub-second Duration only overflows after ~584 billion years of process uptime"
            )]
            let new_due = due + tick;

            next_due = Some(new_due);
        }

        self.next_frame_due = next_due;

        Ok(())
    }

    /// Writes any audio captured since the last drain (or since the event
    /// started, for the first call) into the temp PCM file. Must be polled
    /// periodically while the event is active so live audio keeps
    /// accumulating for the clip's full duration, not just the pre-buffer
    /// window written in `start`.
    pub fn drain_audio(
        &mut self,
        ring_buffer: &std::sync::Mutex<crate::buffer::RingBuffer>,
    ) -> Result<()> {
        let new_chunks = {
            let buf = ring_buffer.lock().expect("ring buffer lock poisoned");
            buf.audio_since(self.last_audio_drain_at)
        };

        if let Some(last) = new_chunks.last() {
            self.last_audio_drain_at = last.timestamp;
        }

        for chunk in &new_chunks {
            self.write_audio(&chunk.samples)?;
        }

        Ok(())
    }

    /// Streams one raw video frame to ffmpeg's stdin.
    pub fn write_frame(&mut self, image: &image::RgbImage) -> Result<()> {
        let stdin = self
            .ffmpeg_video
            .stdin
            .as_mut()
            .context("ffmpeg stdin was already closed")?;

        stdin
            .write_all(image.as_raw())
            .context("failed to write frame to ffmpeg")?;
        Ok(())
    }

    /// Appends raw PCM samples to the temp audio file.
    pub fn write_audio(&mut self, samples: &[f32]) -> Result<()> {
        let mut bytes = Vec::with_capacity(samples.len().saturating_mul(size_of::<f32>()));

        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }

        self.audio_file
            .write_all(&bytes)
            .context("failed to write audio samples to temp file")?;
        Ok(())
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
        let offset_secs = frame_timestamp
            .saturating_duration_since(self.clip_timeline_start)
            .as_secs_f64();

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

    /// How long it's been since the last trigger/touch.
    pub fn quiet_for(&self) -> Duration {
        self.last_trigger_at.elapsed()
    }

    /// Stops encoding, muxes the buffered audio into the video, and writes
    /// the JSON sidecar alongside the final clip.
    pub fn finish(mut self) -> Result<()> {
        drop(self.ffmpeg_video.stdin.take());

        let mut stderr = String::new();

        if let Some(mut stderr_pipe) = self.ffmpeg_video.stderr.take() {
            let _ = std::io::Read::read_to_string(&mut stderr_pipe, &mut stderr);
        }

        let status = self
            .ffmpeg_video
            .wait()
            .context("ffmpeg video encoder failed")?;

        if !status.success() {
            bail!(
                "ffmpeg video encoder exited with {status}: {}",
                stderr.trim()
            );
        }

        drop(self.audio_file);

        mux_audio_into_video(
            &self.video_tmp_path,
            &self.audio_tmp_path,
            &self.final_clip_path,
            self.audio_sample_rate,
            self.audio_channels,
        )?;

        let _ = std::fs::remove_file(&self.video_tmp_path);
        let _ = std::fs::remove_file(&self.audio_tmp_path);

        let all_classes: Vec<&str> = self.all_classes.iter().copied().collect();
        let renamed_path = clip_path(&self.output_dir, self.started_at, &all_classes)?;

        if renamed_path != self.final_clip_path {
            std::fs::rename(&self.final_clip_path, &renamed_path).with_context(|| {
                format!(
                    "failed to rename clip {} to {} (reflecting every class detected during the clip, per ADR 4)",
                    self.final_clip_path.display(),
                    renamed_path.display()
                )
            })?;
            self.final_clip_path = renamed_path;
        }

        let sidecar = Sidecar {
            detections: self.detections,
        };

        let sidecar_json =
            serde_json::to_string_pretty(&sidecar).context("failed to serialize sidecar")?;

        std::fs::write(sidecar_path(&self.final_clip_path), sidecar_json)
            .context("failed to write sidecar file")?;

        Ok(())
    }
}

/// Spawns ffmpeg to encode raw RGB frames fed via stdin into an H.264 file.
fn spawn_video_encoder(
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
    // doesn't reach it directly -- it shares the foreground process group
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
fn mux_audio_into_video(
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
fn resample_to_frame_rate(frames: &[TimestampedFrame], frame_rate: u32) -> Vec<&TimestampedFrame> {
    let Some(first) = frames.first() else {
        return Vec::new();
    };

    if frame_rate == 0 {
        return frames.iter().collect();
    }

    let tick = Duration::from_secs_f64(1.0 / f64::from(frame_rate));
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
