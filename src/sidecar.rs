use serde::Serialize;

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

#[cfg(test)]
mod tests {
    //! Unit test for `Sidecar` JSON serialization.
    #![allow(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        clippy::missing_panics_doc,
        reason = "test assertions favor unwrap/indexing for clarity; panics here fail the test, which is the intended behavior"
    )]

    use super::*;

    /// Fixture detection offset, in seconds from the clip start.
    const DETECTION_OFFSET_SECS: f64 = 1.5;

    /// Fixture detection confidence.
    const DETECTION_CONFIDENCE: f32 = 0.87;

    /// Fixture motion-event offset, in seconds from the clip start.
    const MOTION_OFFSET_SECS: f64 = 0.2;

    /// Fixture motion-event changed-pixel ratio.
    const MOTION_CHANGED_RATIO: f32 = 0.05;

    /// Tolerance for comparing a round-tripped JSON float against its
    /// original fixture value.
    const JSON_FLOAT_TOLERANCE: f64 = 0.001;

    #[test]
    fn sidecar_serializes_with_expected_shape() {
        let sidecar = Sidecar {
            detections: vec![DetectionRecord {
                offset_secs: DETECTION_OFFSET_SECS,
                class_name: "person".to_string(),
                confidence: DETECTION_CONFIDENCE,
            }],
            motion_events: vec![MotionEvent {
                offset_secs: MOTION_OFFSET_SECS,
                changed_ratio: MOTION_CHANGED_RATIO,
            }],
        };

        let json = serde_json::to_value(&sidecar).unwrap();

        assert_eq!(json["detections"][0]["class_name"], "person");
        assert!(
            (json["detections"][0]["offset_secs"].as_f64().unwrap() - DETECTION_OFFSET_SECS).abs()
                < JSON_FLOAT_TOLERANCE
        );
        assert!(
            (json["motion_events"][0]["changed_ratio"].as_f64().unwrap()
                - f64::from(MOTION_CHANGED_RATIO))
            .abs()
                < JSON_FLOAT_TOLERANCE
        );
    }
}
