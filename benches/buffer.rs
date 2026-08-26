//! Benchmarks for `RingBuffer` push/read throughput, standing in for the
//! detection/writer/preview loops' hot paths (`push_frame` runs on every
//! captured camera frame; `latest_frame`/`frames_since` run on every
//! detection/writer poll). See `docs/adr/0007-testing-conventions.md` for
//! why this crate needs a library target at all to make `benches/` possible.
#![allow(
    missing_docs,
    clippy::missing_docs_in_private_items,
    reason = "criterion_group!/criterion_main! generate an undocumented fn main; this file has \
               no public API of its own for missing_docs to meaningfully enforce"
)]
use std::hint::black_box;
use std::time::{Duration, Instant};

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use image::RgbImage;
use motioncap::buffer::RingBuffer;

/// A generous retention window so prefilled frames never evict mid-benchmark
/// in the read-path benchmarks (`latest_frame`, `frames_since`), which push
/// only a small, fixed number of frames before the timed portion runs.
const BENCH_RETENTION: Duration = Duration::from_mins(1);

/// A short retention window for the `push_frame` benchmark specifically, so
/// each push evicts the frame from several pushes ago instead of retaining
/// every frame for the whole run. Criterion's default warm-up + measurement
/// window runs for several seconds, during which `push_frame` (tens of
/// nanoseconds per call) executes many millions of times; without eviction
/// actually firing, `BENCH_RETENTION`'s one-minute window would let a single
/// invocation accumulate millions of unevicted 640x480 frames (gigabytes of
/// image data) and measure ever-growing `VecDeque` reallocation cost instead
/// of the bounded, steady-state append+evict cost `push_frame` actually has
/// in production.
const PUSH_FRAME_BENCH_RETENTION: Duration = Duration::from_millis(100);

/// Representative camera frame dimensions (matches this project's webcam).
const FRAME_WIDTH: u32 = 640;
/// Representative camera frame dimensions (matches this project's webcam).
const FRAME_HEIGHT: u32 = 480;

/// Number of frames pre-populated before a read-path benchmark runs, so
/// `latest_frame` has a realistic backlog to scan past.
const PREFILLED_FRAME_COUNT: usize = 64;

/// Number of trailing prefilled frames that a `frames_since` call should
/// actually return, standing in for the small handful of frames a real
/// per-poll drain (`RecordingEvent::drain_frames`/`drain_audio`, polled at
/// `RECORDING_FRAME_RATE`) sees since its own last poll, not the entire
/// retained backlog.
const RECENT_FRAME_COUNT: usize = 4;

fn push_frame(c: &mut Criterion) {
    let mut buffer = RingBuffer::new(PUSH_FRAME_BENCH_RETENTION);
    let frame = RgbImage::new(FRAME_WIDTH, FRAME_HEIGHT);

    c.bench_function("ring_buffer_push_frame", |b| {
        b.iter_batched(
            || frame.clone(),
            |owned_frame| buffer.push_frame(black_box(owned_frame)),
            BatchSize::SmallInput,
        );
    });
}

fn latest_frame(c: &mut Criterion) {
    let mut buffer = RingBuffer::new(BENCH_RETENTION);
    let frame = RgbImage::new(FRAME_WIDTH, FRAME_HEIGHT);

    for _ in 0..PREFILLED_FRAME_COUNT {
        buffer.push_frame(frame.clone());
    }

    c.bench_function("ring_buffer_latest_frame", |b| {
        b.iter(|| black_box(buffer.latest_frame()));
    });
}

fn frames_since(c: &mut Criterion) {
    let mut buffer = RingBuffer::new(BENCH_RETENTION);
    let frame = RgbImage::new(FRAME_WIDTH, FRAME_HEIGHT);

    for _ in 0..(PREFILLED_FRAME_COUNT - RECENT_FRAME_COUNT) {
        buffer.push_frame(frame.clone());
    }

    let since = Instant::now();

    for _ in 0..RECENT_FRAME_COUNT {
        buffer.push_frame(frame.clone());
    }

    c.bench_function("ring_buffer_frames_since", |b| {
        b.iter(|| black_box(buffer.frames_since(since)));
    });
}

criterion_group!(benches, push_frame, latest_frame, frames_since);
criterion_main!(benches);
