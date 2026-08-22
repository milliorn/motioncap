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
}
