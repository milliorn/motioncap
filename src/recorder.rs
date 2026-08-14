use std::collections::BTreeSet;
use std::io::Write;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Local};
use serde::Serialize;

use crate::buffer::{TimestampedAudio, TimestampedFrame};
use crate::paths::{clip_path, sidecar_path};

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

/// One recorded motion-gate trip, written into a clip's `.json` sidecar.
/// Logged for every trip during an active recording, whether or not YOLO
/// went on to confirm a living-thing class that same tick; this is what
/// lets a clip that kept extending (via the post-buffer quiet window) be
/// audited after the fact for what actually kept triggering it.
#[derive(Serialize)]
pub struct MotionEvent {
    /// Seconds from the start of the clip when the gate tripped.
    pub offset_secs: f64,
    /// Fraction of pixels (0.0-1.0) the background model marked as changed.
    pub changed_ratio: f32,
}

/// A clip's `.json` sidecar contents (ADR 4).
#[derive(Serialize)]
pub struct Sidecar {
    /// Every detection recorded during the clip, in chronological order.
    pub detections: Vec<DetectionRecord>,
    /// Every motion-gate trip recorded during the clip, in chronological
    /// order, including ones YOLO never confirmed as a living thing.
    pub motion_events: Vec<MotionEvent>,
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

/// Pure timing/bookkeeping state for a single recorded clip, split out from
/// `RecordingEvent` so it can be unit-tested by constructing a bare
/// `ClipState` directly, with no ffmpeg process or temp files involved. Holds
/// everything about a clip's lifecycle *except* the actual I/O handles
/// (`RecordingEvent`'s `ffmpeg_video`/`audio_file`/temp-file paths), which
/// `RecordingEvent` still owns and drives through this struct's methods.
struct ClipState {
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
    /// Wall-clock time a real, camera-delivered frame was last written. Used
    /// by `camera_stalled` to detect when the camera has stopped delivering
    /// frames. A security recording must never paper over a gap by
    /// fabricating footage, so unlike a naive resampler this never duplicates
    /// a frame to fill a missed tick; it reports the stall instead.
    last_real_frame_at: Instant,
    /// Every distinct class detected so far during this clip (ADR 4: the
    /// final filename must reflect every class seen over the clip's
    /// lifetime, not just the classes that triggered it).
    all_classes: BTreeSet<&'static str>,
    /// Every detection recorded so far during this clip.
    detections: Vec<DetectionRecord>,
    /// Every motion-gate trip recorded so far during this clip.
    motion_events: Vec<MotionEvent>,
}

impl ClipState {
    /// Starts fresh bookkeeping for a clip beginning now, with playback at `frame_rate`.
    fn new(frame_rate: u32, clip_timeline_start: Instant) -> Self {
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
    fn camera_stalled(&self) -> bool {
        self.last_real_frame_at.elapsed() >= MAX_FRAME_STALL
    }

    /// Duration of one frame-rate tick at this event's configured `frame_rate`.
    fn frame_tick(&self) -> Duration {
        Duration::from_secs_f64(1.0 / f64::from(self.frame_rate))
    }

    /// Decides whether a newly-drained frame at `frame_timestamp` is due to
    /// be written at this event's configured `frame_rate`, advancing
    /// `next_frame_due` to the following tick if so. The first call with no
    /// prior `next_frame_due` treats `frame_timestamp` itself as due, so
    /// draining never stalls waiting for a tick boundary that predates the
    /// first frame it ever saw.
    fn should_write_frame(&mut self, frame_timestamp: Instant) -> bool {
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
    fn anchor_next_tick_after_seed(&mut self, last_seeded_timestamp: Instant) {
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
    fn record_detection(
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
    fn touch(&mut self) {
        self.last_trigger_at = Instant::now();
    }

    /// Records a motion-gate trip into the sidecar for later audit (e.g.
    /// distinguishing a fan/curtain repeatedly tripping the gate from an
    /// actual subject). Purely diagnostic bookkeeping. It does not itself
    /// touch the post-buffer quiet window. Callers already call
    /// `touch`/`record_detection` for that as needed.
    fn record_motion(&mut self, changed_ratio: f32, frame_timestamp: Instant) {
        let offset_secs = self.offset_secs(frame_timestamp);

        self.motion_events.push(MotionEvent {
            offset_secs,
            changed_ratio,
        });
    }

    /// Seconds from the clip's actual timeline start (not wall-clock time at
    /// the moment the caller happens to run) to `frame_timestamp`, the
    /// capture timestamp of the frame being recorded against.
    fn offset_secs(&self, frame_timestamp: Instant) -> f64 {
        frame_timestamp
            .saturating_duration_since(self.clip_timeline_start)
            .as_secs_f64()
    }

    /// How long it's been since the last trigger/touch.
    fn quiet_for(&self) -> Duration {
        self.last_trigger_at.elapsed()
    }
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
    /// Pure timing/bookkeeping state for this clip (see `ClipState`).
    state: ClipState,
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
    /// Deliberately does *not* write the pre-buffer here. See `seed`.
    /// Writing dozens of frames synchronously on the caller's thread would
    /// block it for long enough that real wall-clock time passes with
    /// nothing being recorded, which shows up as a skip/jump right at the
    /// start of the clip once live writing resumes. Callers should invoke
    /// `seed` from whatever thread is responsible for steady-paced frame
    /// writing instead.
    ///
    /// `clip_timeline_start` should be the capture timestamp of the earliest
    /// pre-buffer frame that will be seeded into this clip (i.e. the actual
    /// start of the video file), not the trigger instant. It's the zero
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
            state: ClipState::new(frame_rate, clip_timeline_start),
        })
    }

    /// Writes the pre-event buffer (frames captured before the trigger
    /// fired) into a freshly-started event. Must be called once, as the
    /// first write against a new event, before any live `write_frame`/
    /// `drain_audio` calls. See the note on `start` for why this is
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
        let selected = resample_to_frame_rate(pre_frames, self.state.frame_rate);

        for frame in &selected {
            self.write_frame(&frame.image)?;
        }

        if !selected.is_empty() {
            self.state.last_real_frame_at = Instant::now();
        }

        if let Some(last) = selected.last() {
            self.state.anchor_next_tick_after_seed(last.timestamp);
        }

        for chunk in pre_audio {
            self.write_audio(&chunk.samples)?;
        }

        if let Some(last) = pre_audio.last() {
            self.state.last_audio_drain_at = last.timestamp;
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
    /// This is a security recording: it must never paper over a gap in
    /// coverage by fabricating footage. If the camera doesn't deliver a real
    /// frame for a tick that's already due by wall-clock `Instant::now()`,
    /// this does *not* duplicate the last frame to fill it. It stops
    /// writing and leaves `camera_stalled` reporting true so the caller ends
    /// the recording instead of the clip silently containing footage that
    /// was never actually captured.
    pub fn drain_frames(
        &mut self,
        ring_buffer: &std::sync::Mutex<crate::buffer::RingBuffer>,
    ) -> Result<()> {
        let new_frames = {
            let buf = ring_buffer.lock().expect("ring buffer lock poisoned");
            buf.frames_since(self.state.last_frame_drain_at)
        };

        if let Some(last) = new_frames.last() {
            self.state.last_frame_drain_at = last.timestamp;
        }

        if !new_frames.is_empty() {
            self.state.last_real_frame_at = Instant::now();
        }

        for frame in &new_frames {
            if self.state.should_write_frame(frame.timestamp) {
                self.write_frame(&frame.image)?;
            }
        }

        Ok(())
    }

    /// True once the camera has gone `MAX_FRAME_STALL` without delivering a
    /// real frame. Callers should treat this as a signal to end the
    /// recording: `drain_frames` never fabricates a frame to cover for the
    /// camera, and the motion gate has nothing new to evaluate once frames
    /// stop arriving, so nothing else will naturally close the clip.
    pub fn camera_stalled(&self) -> bool {
        self.state.camera_stalled()
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
            buf.audio_since(self.state.last_audio_drain_at)
        };

        if let Some(last) = new_chunks.last() {
            self.state.last_audio_drain_at = last.timestamp;
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
        self.state
            .record_detection(class_name, confidence, frame_timestamp);
    }

    /// Resets the post-buffer quiet window without recording a new detection
    /// (e.g. motion continues but wasn't re-confirmed by YOLO this tick).
    pub fn touch(&mut self) {
        self.state.touch();
    }

    /// Records a motion-gate trip into the sidecar for later audit (e.g.
    /// distinguishing a fan/curtain repeatedly tripping the gate from an
    /// actual subject). Purely diagnostic bookkeeping. It does not itself
    /// touch the post-buffer quiet window. Callers already call
    /// `touch`/`record_detection` for that as needed.
    pub fn record_motion(&mut self, changed_ratio: f32, frame_timestamp: Instant) {
        self.state.record_motion(changed_ratio, frame_timestamp);
    }

    /// How long it's been since the last trigger/touch.
    pub fn quiet_for(&self) -> Duration {
        self.state.quiet_for()
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

        let all_classes: Vec<&str> = self.state.all_classes.iter().copied().collect();
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
            detections: self.state.detections,
            motion_events: self.state.motion_events,
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

#[cfg(test)]
mod tests {
    //! Unit tests for `ClipState`'s bookkeeping (no ffmpeg process required),
    //! `resample_to_frame_rate`, and sidecar JSON serialization, plus inline
    //! ffmpeg-backed round-trip tests for `RecordingEvent::start`/`seed`/
    //! `drain_frames`/`drain_audio`/`finish` further down in this same module
    //! (no top-level `tests/` directory exists for this binary-only crate, see ADR 7).
    #![allow(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        clippy::missing_panics_doc,
        clippy::unchecked_time_subtraction,
        reason = "test assertions favor unwrap/indexing/plain time arithmetic for clarity; test \
                   durations are small hardcoded constants, so underflow is not reachable"
    )]

    use super::*;

    fn frame_at(timestamp: Instant) -> TimestampedFrame {
        TimestampedFrame {
            timestamp,
            image: image::RgbImage::new(1, 1),
        }
    }

    // --- ClipState ---

    #[test]
    fn camera_stalled_false_when_recently_touched() {
        let state = ClipState::new(15, Instant::now());
        assert!(!state.camera_stalled());
    }

    #[test]
    fn camera_stalled_true_once_max_frame_stall_elapses() {
        let mut state = ClipState::new(15, Instant::now());
        state.last_real_frame_at = Instant::now() - MAX_FRAME_STALL - Duration::from_millis(50);
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

    // --- resample_to_frame_rate ---

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

    // --- Sidecar serde round-trip ---

    #[test]
    fn sidecar_serializes_with_expected_shape() {
        let sidecar = Sidecar {
            detections: vec![DetectionRecord {
                offset_secs: 1.5,
                class_name: "person".to_string(),
                confidence: 0.87,
            }],
            motion_events: vec![MotionEvent {
                offset_secs: 0.2,
                changed_ratio: 0.05,
            }],
        };

        let json = serde_json::to_value(&sidecar).unwrap();

        assert_eq!(json["detections"][0]["class_name"], "person");
        assert!((json["detections"][0]["offset_secs"].as_f64().unwrap() - 1.5).abs() < 0.001);
        assert!((json["motion_events"][0]["changed_ratio"].as_f64().unwrap() - 0.05).abs() < 0.001);
    }

    // --- RecordingEvent end-to-end (real ffmpeg subprocess) ---

    /// Runs `ffprobe` against `path` and returns its parsed JSON stream info,
    /// to confirm the muxed output is a real, playable file rather than just
    /// a file that happens to exist on disk.
    fn ffprobe_streams(path: &std::path::Path) -> serde_json::Value {
        let output = Command::new("ffprobe")
            .args([
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
            ])
            .arg(path)
            .output()
            .expect("failed to spawn ffprobe");
        assert!(
            output.status.success(),
            "ffprobe failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice(&output.stdout).expect("ffprobe produced invalid JSON")
    }

    /// A 2x2 timestamped frame; libx264 requires even width/height, so this
    /// (rather than the shared `frame_at`'s 1x1) is used for the ffmpeg
    /// round-trip test below.
    fn even_sized_frame_at(timestamp: Instant) -> TimestampedFrame {
        TimestampedFrame {
            timestamp,
            image: image::RgbImage::new(2, 2),
        }
    }

    #[test]
    fn recording_event_round_trip_produces_valid_mp4_and_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let clip_timeline_start = Instant::now();
        let started_at = chrono::Local::now();

        // finish() recomputes the final filename from the full accumulated
        // class list (ADR 4) via the same clip_path() helper; the initial
        // final_clip_path only matters as the pre-rename name, so predict the
        // real final path with clip_path() up front rather than guessing it.
        let initial_path = clip_path(dir.path(), started_at, &[]).unwrap();
        let expected_final_path = clip_path(dir.path(), started_at, &["person"]).unwrap();

        let mut event = RecordingEvent::start(RecordingEventParams {
            final_clip_path: initial_path,
            output_dir: dir.path().to_path_buf(),
            started_at,
            width: 2,
            height: 2,
            frame_rate: 5,
            audio_sample_rate: 8000,
            audio_channels: 1,
            clip_timeline_start,
        })
        .unwrap();

        let pre_frames = vec![
            even_sized_frame_at(clip_timeline_start),
            even_sized_frame_at(clip_timeline_start + Duration::from_millis(250)),
        ];
        let pre_audio = vec![TimestampedAudio {
            timestamp: clip_timeline_start,
            samples: vec![0.0; 800],
        }];

        event.seed(&pre_frames, &pre_audio).unwrap();

        event.record_motion(0.02, clip_timeline_start);
        event.record_detection("person", 0.95, clip_timeline_start);
        assert!(event.quiet_for() < Duration::from_secs(1));
        event.touch();

        event.finish().unwrap();

        assert!(
            expected_final_path.exists(),
            "expected clip at {}",
            expected_final_path.display()
        );

        let sidecar_json = std::fs::read_to_string(sidecar_path(&expected_final_path)).unwrap();
        let sidecar: serde_json::Value = serde_json::from_str(&sidecar_json).unwrap();
        assert_eq!(sidecar["detections"].as_array().unwrap().len(), 1);
        assert_eq!(sidecar["motion_events"].as_array().unwrap().len(), 1);
        assert_eq!(sidecar["detections"][0]["class_name"], "person");

        let probe = ffprobe_streams(&expected_final_path);
        let streams = probe["streams"].as_array().unwrap();
        assert!(
            streams.iter().any(|s| s["codec_type"] == "video"),
            "expected a video stream in {probe}"
        );
        assert!(
            streams.iter().any(|s| s["codec_type"] == "audio"),
            "expected an audio stream in {probe}"
        );
    }

    #[test]
    fn drain_frames_and_drain_audio_write_newly_buffered_content() {
        let dir = tempfile::tempdir().unwrap();
        let clip_timeline_start = Instant::now();
        let started_at = chrono::Local::now();

        let initial_path = clip_path(dir.path(), started_at, &[]).unwrap();

        let mut event = RecordingEvent::start(RecordingEventParams {
            final_clip_path: initial_path,
            output_dir: dir.path().to_path_buf(),
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
                &[even_sized_frame_at(clip_timeline_start)],
                &[TimestampedAudio {
                    timestamp: clip_timeline_start,
                    samples: vec![0.0; 100],
                }],
            )
            .unwrap();

        // At 5fps, seed() sets next_frame_due to clip_timeline_start + 200ms;
        // sleeping past that before pushing ensures the newly-buffered frame
        // is actually due and gets written (not silently skipped for arriving
        // before its tick), exercising drain_frames' write branch for real.
        std::thread::sleep(Duration::from_millis(250));

        let ring_buffer =
            std::sync::Mutex::new(crate::buffer::RingBuffer::new(Duration::from_secs(30)));
        {
            let mut buf = ring_buffer.lock().unwrap();
            buf.push_frame(image::RgbImage::new(2, 2));
            buf.push_audio(vec![0.5; 100]);
        }

        // Both must succeed without error against real newly-buffered content
        // pushed after seed(); this is the steady-poll path the recording
        // writer thread drives in production.
        event.drain_frames(&ring_buffer).unwrap();
        event.drain_audio(&ring_buffer).unwrap();

        assert!(!event.camera_stalled());

        event.finish().unwrap();
    }
}
