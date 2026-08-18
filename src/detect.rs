use std::fmt::Debug;

use anyhow::Result;
use image::{RgbImage, imageops::FilterType};
use ndarray::{Array4, s};
use ort::value::Shape;

/// `ort::Error` doesn't implement `std::error::Error` in a way `anyhow::Context`
/// can use directly, so ort results are converted through this helper instead.
pub fn ort_err<T, E: Debug>(result: std::result::Result<T, E>, msg: &str) -> Result<T> {
    result.map_err(|e| anyhow::anyhow!("{msg}: {e:?}"))
}

/// Serializes every `#[ignore]`d test across the crate that constructs a real
/// `Detector`/ONNX Runtime session and/or a real `MotionGate` (both here and
/// in `main.rs`'s test module). `cargo test`'s default parallelism ran
/// multiple such tests concurrently and reliably produced a heap-corruption
/// abort ("corrupted double-linked list") within a handful of runs. Neither
/// `ort`'s `Session` nor `OpenCV`'s `BackgroundSubtractorMOG2` are documented
/// as safe to construct/run concurrently across independent instances in
/// separate threads, and this crate's production code never does so (YOLO
/// inference and the motion gate both run single-threaded inside the
/// detection worker). Every `#[ignore]`d test that touches either must
/// acquire this lock for its full duration before doing anything else.
#[cfg(test)]
pub static MODEL_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// COCO class indices for "person" plus every animal class, per the
/// "every living thing" detection scope decided during planning (ADR 2).
const LIVING_THING_CLASSES: &[(usize, &str)] = &[
    (0, "person"),
    (14, "bird"),
    (15, "cat"),
    (16, "dog"),
    (17, "horse"),
    (18, "sheep"),
    (19, "cow"),
    (20, "elephant"),
    (21, "bear"),
    (22, "zebra"),
    (23, "giraffe"),
];

/// `YOLOv8`'s fixed square input resolution.
const MODEL_INPUT_SIZE: u32 = 640;

/// A single YOLO detection above the confidence threshold.
pub struct Detection {
    /// The detected COCO class name (person or an animal, see `LIVING_THING_CLASSES`).
    pub class_name: &'static str,
    /// The model's reported confidence for this detection.
    pub confidence: f32,
}

/// Letterboxes the frame into a square canvas (preserving aspect ratio, padding
/// with grey) and converts to an NCHW f32 tensor normalized to [0, 1]. This is the
/// standard `YOLOv8` preprocessing; a naive stretch-to-square resize distorts
/// non-square camera frames (e.g. this webcam's 640x480) enough to
/// meaningfully degrade detection confidence, since the model is trained on
/// letterboxed inputs, not stretched ones.
#[allow(
    clippy::cast_precision_loss,
    reason = "camera frame dimensions never approach f32's 24-bit exact-integer range"
)]
pub fn preprocess(frame: &RgbImage) -> Array4<f32> {
    let (src_w, src_h) = frame.dimensions();
    let scale =
        (MODEL_INPUT_SIZE as f32 / src_w as f32).min(MODEL_INPUT_SIZE as f32 / src_h as f32);
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "scale is derived from MODEL_INPUT_SIZE / src_dim, so result stays within [0, MODEL_INPUT_SIZE]"
    )]
    let new_w = (src_w as f32 * scale).round() as u32;
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "scale is derived from MODEL_INPUT_SIZE / src_dim, so result stays within [0, MODEL_INPUT_SIZE]"
    )]
    let new_h = (src_h as f32 * scale).round() as u32;

    let resized = image::imageops::resize(frame, new_w, new_h, FilterType::Triangle);

    // new_w/new_h <= MODEL_INPUT_SIZE always holds: scale = min(SIZE/src_w,
    // SIZE/src_h) <= 1.0, so new_w = round(src_w * scale) <= SIZE (likewise
    // new_h). The subtraction and the pad_x/pad_y additions below therefore
    // never overflow, and the resulting tensor index is always < SIZE,
    // matching tensor's allocated shape.
    #[allow(
        clippy::arithmetic_side_effects,
        reason = "new_w/new_h <= MODEL_INPUT_SIZE, proven above"
    )]
    let pad_x = (MODEL_INPUT_SIZE - new_w) / 2;
    #[allow(
        clippy::arithmetic_side_effects,
        reason = "new_w/new_h <= MODEL_INPUT_SIZE, proven above"
    )]
    let pad_y = (MODEL_INPUT_SIZE - new_h) / 2;

    let mut tensor = Array4::<f32>::from_elem(
        (1, 3, MODEL_INPUT_SIZE as usize, MODEL_INPUT_SIZE as usize),
        114.0 / 255.0,
    );
    for (x, y, pixel) in resized.enumerate_pixels() {
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "x < new_w and pad_x + new_w <= MODEL_INPUT_SIZE, so x + pad_x never overflows"
        )]
        let dst_x = (x + pad_x) as usize;
        #[allow(
            clippy::arithmetic_side_effects,
            reason = "y < new_h and pad_y + new_h <= MODEL_INPUT_SIZE, so y + pad_y never overflows"
        )]
        let dst_y = (y + pad_y) as usize;
        #[allow(
            clippy::indexing_slicing,
            reason = "dst_x, dst_y < MODEL_INPUT_SIZE, matching tensor's allocated shape"
        )]
        {
            tensor[[0, 0, dst_y, dst_x]] = f32::from(pixel[0]) / 255.0;
            tensor[[0, 1, dst_y, dst_x]] = f32::from(pixel[1]) / 255.0;
            tensor[[0, 2, dst_y, dst_x]] = f32::from(pixel[2]) / 255.0;
        }
    }
    tensor
}

/// Parses `YOLOv8`'s standard `[1, 84, 8400]` output (4 box coords + 80 class
/// scores, per anchor) and filters to living-thing classes above threshold.
/// No NMS/box deduplication is performed since only class presence (not
/// precise box geometry) is needed to decide whether to trigger a recording.
pub fn postprocess(shape: &Shape, data: &[f32], confidence_threshold: f32) -> Vec<Detection> {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "ONNX tensor dims are always small positive values (e.g. 1, 84, 8400)"
    )]
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

    let [_, class_dim, num_anchors] = dims[..] else {
        log::warn!("unexpected YOLO output shape: {dims:?}");
        return Vec::new();
    };
    if class_dim != 84 {
        log::warn!("unexpected YOLO output shape: {dims:?}");
        return Vec::new();
    }

    let view = ndarray::ArrayView2::from_shape((84, num_anchors), data)
        .expect("output tensor size should match its own reported shape");

    let mut best_per_class: std::collections::HashMap<usize, f32> =
        std::collections::HashMap::new();

    for anchor in 0..num_anchors {
        let scores = view.slice(s![4..84, anchor]);
        for (class_idx, &score) in scores.iter().enumerate() {
            if score < confidence_threshold {
                continue;
            }
            best_per_class
                .entry(class_idx)
                .and_modify(|best| {
                    if score > *best {
                        *best = score;
                    }
                })
                .or_insert(score);
        }
    }

    LIVING_THING_CLASSES
        .iter()
        .filter_map(|&(class_idx, name)| {
            best_per_class.get(&class_idx).map(|&confidence| Detection {
                class_name: name,
                confidence,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    //! Unit tests for `preprocess`/`postprocess`/`ort_err`, exercised without a
    //! real ONNX Runtime session or model file.
    #![allow(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        clippy::missing_panics_doc,
        clippy::arithmetic_side_effects,
        clippy::cast_possible_wrap,
        reason = "test assertions favor unwrap/indexing/plain arithmetic for clarity over the \
                   production-code proof-obligation style used elsewhere in this crate; test \
                   data sizes are small hardcoded constants, so overflow/wrap is not reachable"
    )]

    use image::Rgb;
    use ort::value::Shape;

    use super::*;

    /// Builds a synthetic `[1, 84, num_anchors]` YOLO output buffer where every
    /// anchor's class scores are zero except the ones set via `scores`, a list
    /// of `(anchor, class_idx, score)` triples (`class_idx` is 0-based over the
    /// 80 COCO classes, matching `LIVING_THING_CLASSES`' indices).
    fn synthetic_output(num_anchors: usize, scores: &[(usize, usize, f32)]) -> Vec<f32> {
        let mut data = vec![0.0_f32; 84 * num_anchors];
        for &(anchor, class_idx, score) in scores {
            let row = 4 + class_idx;
            data[row * num_anchors + anchor] = score;
        }
        data
    }

    #[test]
    fn postprocess_filters_to_living_thing_classes_above_threshold() {
        let num_anchors = 2;
        // class_idx 0 = person (living-thing, above threshold);
        // class_idx 1 = bicycle (not in LIVING_THING_CLASSES, ignored).
        let data = synthetic_output(num_anchors, &[(0, 0, 0.9), (1, 1, 0.9)]);
        let shape = Shape::from(vec![1_i64, 84, num_anchors as i64]);

        let detections = postprocess(&shape, &data, 0.3);

        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].class_name, "person");
        assert!((detections[0].confidence - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn postprocess_excludes_scores_below_threshold() {
        let num_anchors = 1;
        let data = synthetic_output(num_anchors, &[(0, 0, 0.1)]);
        let shape = Shape::from(vec![1_i64, 84, num_anchors as i64]);

        let detections = postprocess(&shape, &data, 0.3);

        assert!(detections.is_empty());
    }

    #[test]
    fn postprocess_keeps_best_score_per_class_across_anchors() {
        let num_anchors = 2;
        let data = synthetic_output(num_anchors, &[(0, 0, 0.4), (1, 0, 0.95)]);
        let shape = Shape::from(vec![1_i64, 84, num_anchors as i64]);

        let detections = postprocess(&shape, &data, 0.3);

        assert_eq!(detections.len(), 1);
        assert!((detections[0].confidence - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn postprocess_returns_empty_on_wrong_class_dim() {
        let shape = Shape::from(vec![1_i64, 85, 10_i64]);
        let data = vec![0.0_f32; 85 * 10];

        assert!(postprocess(&shape, &data, 0.3).is_empty());
    }

    #[test]
    fn postprocess_returns_empty_on_wrong_rank() {
        let shape = Shape::from(vec![84_i64, 10_i64]);
        let data = vec![0.0_f32; 84 * 10];

        assert!(postprocess(&shape, &data, 0.3).is_empty());
    }

    #[test]
    fn preprocess_pads_vertically_for_landscape_input() {
        let frame = RgbImage::from_pixel(640, 480, Rgb([200, 200, 200]));
        let tensor = preprocess(&frame);

        // Landscape (wider than tall): scale is bound by width, so the resized
        // image is shorter than MODEL_INPUT_SIZE and padding is added on the
        // vertical axis only. The top row should be the grey pad fill, not a
        // resized pixel.
        let pad_value = 114.0 / 255.0;
        assert!((tensor[[0, 0, 0, 0]] - pad_value).abs() < f32::EPSILON);
    }

    #[test]
    fn preprocess_pads_horizontally_for_portrait_input() {
        let frame = RgbImage::from_pixel(480, 640, Rgb([200, 200, 200]));
        let tensor = preprocess(&frame);

        let pad_value = 114.0 / 255.0;
        assert!((tensor[[0, 0, 0, 0]] - pad_value).abs() < f32::EPSILON);
    }

    #[test]
    fn preprocess_adds_no_padding_for_square_input() {
        let frame = RgbImage::from_pixel(640, 640, Rgb([10, 20, 30])); // deliberately non-grey
        let tensor = preprocess(&frame);

        // A square input needs no letterboxing: scale = 1.0, new_w = new_h =
        // MODEL_INPUT_SIZE, so every pixel in the tensor comes from the
        // resized image rather than the grey pad fill.
        let pad_value = 114.0 / 255.0;
        assert!((tensor[[0, 0, 0, 0]] - pad_value).abs() > 0.05);
    }

    #[test]
    fn ort_err_passes_through_ok() {
        let result: Result<u32, &str> = Ok(42);
        assert_eq!(ort_err(result, "unused").unwrap(), 42);
    }

    #[test]
    fn ort_err_wraps_error_with_message() {
        let result: Result<u32, &str> = Err("boom");
        let err = ort_err(result, "context message").unwrap_err();
        let text = format!("{err}");
        assert!(text.contains("context message"));
        assert!(text.contains("boom"));
    }
}
