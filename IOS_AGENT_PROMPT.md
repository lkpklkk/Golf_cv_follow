# iOS Golf Swing Recognizer Port

Build a minimal native iOS SwiftUI app that runs the existing golf action classifier on live camera video. Do not retrain the model or change the feature schema. The repository already contains these generated artifacts:

- `weights/action_sequence_classifier.mlpackage` — preferred Core ML model for Xcode
- `weights/action_sequence_classifier.onnx` — reference/export fallback
- `weights/action_sequence_classifier.metadata.json` — machine-readable contract
- `action/preprocessing.py` — reference implementation of preprocessing

Add the `.mlpackage` to the Xcode target. The generated model should expose:

- Input: `pose_features`, Float32, shape `[1, 30, 17, 19]`
- Output: `logits`, Float32, shape `[1, 3]`
- Labels: class ID `0 = other`, `1 = walk`, `2 = swing`

The model is a temporal Conv1D classifier. It expects 30 pose frames sampled at 15 FPS, representing approximately 2 seconds. The classifier should be invoked every 0.25 seconds after the buffer is warm. Use softmax on the logits. Accept the top class only when its probability is at least `0.75`; otherwise return `other`. The current exported metadata lists a swing threshold of `0.75`, so use that unless making the threshold a clearly labeled app setting.

## Camera and pose pipeline

Use `AVFoundation` for the camera and `Vision` with `VNDetectHumanBodyPoseRequest` for pose estimation. Process frames on a dedicated serial/background queue, not the main thread. Use the most relevant/visible person, or provide a simple person-selection rule if multiple people are present.

The required 17-point COCO order is:

```text
0 nose
1 left eye
2 right eye
3 left ear
4 right ear
5 left shoulder
6 right shoulder
7 left elbow
8 right elbow
9 left wrist
10 right wrist
11 left hip
12 right hip
13 left knee
14 right knee
15 left ankle
16 right ankle
```

Each raw keypoint must be represented as `[xPixels, yPixels, confidence]`, Float32. Use the actual image width and height passed to preprocessing. Vision reports normalized coordinates; verify its coordinate origin and convert to the same top-left pixel convention used by the Python pipeline. In particular, Vision’s normalized y commonly needs to be vertically flipped before converting to pixels:

```swift
xPixels = normalizedX * imageWidth
yPixels = (1.0 - normalizedY) * imageHeight
```

Validate this with an overlay before trusting live predictions.

## Exact preprocessing contract

Port `action/preprocessing.py` faithfully. Do not substitute a generic normalization. For each frame:

1. Convert confidence to `[0, 1]` and mark a joint valid when `confidence >= 0.25`.
2. Compute `frame_xy = [x / width, y / height]`; set invalid joints to zero.
3. Compute body-relative coordinates per frame:
   - Prefer the midpoint of left/right hip (indices 11, 12) as the body center.
   - Otherwise use the midpoint of the shoulders (5, 6).
   - Otherwise use the mean of all valid points.
   - Prefer torso length (distance between shoulder midpoint and hip midpoint) as scale.
   - Otherwise use the maximum x/y span of valid points.
   - Use scale `1.0` if no usable scale exists.
   - `body_xy = (frame_xy - center) / scale`; invalid joints are zero.
4. Compute masked frame velocity: current minus previous `frame_xy`, only when the joint is valid in both frames; first frame is zero.
5. Compute masked body velocity the same way from `body_xy`.
6. Compute these per-frame swing features:
   - left wrist relative distance to left shoulder: norm of `body_xy[9] - body_xy[5]`
   - right wrist relative distance to right shoulder: norm of `body_xy[10] - body_xy[6]`
   - left wrist relative speed: norm of the frame-to-frame difference of that wrist/shoulder vector
   - right wrist relative speed: same for the right side
   - shoulder axis: normalized `body_xy[6] - body_xy[5]`
   - hip axis: normalized `body_xy[12] - body_xy[11]`
   - shoulder rotation delta: `atan2(cross(previousAxis,currentAxis), dot(previousAxis,currentAxis))`
   - hip rotation delta: same for hips
7. For every joint, concatenate these 19 values in exactly this order:

```text
frame_x, frame_y,
body_x, body_y,
confidence,
frame_dx, frame_dy,
body_dx, body_dy,
left_wrist_shoulder_distance,
right_wrist_shoulder_distance,
left_wrist_relative_speed,
right_wrist_relative_speed,
shoulder_axis_x, shoulder_axis_y,
hip_axis_x, hip_axis_y,
shoulder_rotation_delta,
hip_rotation_delta
```

The final tensor is `[30, 17, 19]`, then add a batch dimension to obtain `[1,30,17,19]`. Every invalid joint’s complete feature vector must be zero. Use Float32 throughout. Do not normalize or flatten in a different order: the model’s first operation expects the temporal input to be interpreted as `(batch, time, joint, feature)`.

## Temporal buffer

Implement a timestamped pose buffer equivalent to `action/live_buffer.py`:

- target rate: 15 FPS
- sequence length: 30
- window duration: `(30 - 1) / 15` seconds between first and last sample
- resample each inference window to timestamps `start + i / 15`
- select the nearest captured pose for each target timestamp
- reject/reset if the nearest pose is more than 2.0 seconds away
- reset when the selected person changes
- if a pose is briefly missing, pause and retain history; reset on a genuine gap over 2.0 seconds
- do not classify more often than every 0.25 seconds

Use `CVPixelBuffer`/Vision timestamps rather than UI time. Keep all inference and preprocessing off the main thread.

## Duplicate suppression and recording

The Python classifier returns a prediction for overlapping windows; the iOS app must turn those into one swing event. For `swing` predictions, retain the confidence and window midpoint/time. Apply local-maxima/refractory suppression: within a configurable 1.0-second neighborhood, keep only the highest-confidence swing event. Do not emit repeated events from adjacent overlapping windows. Make this logic unit-testable.

Persist events as Codable JSON. Suggested schema:

```swift
struct SwingEvent: Codable, Identifiable {
    let id: UUID
    let hole: Int
    let shotNumber: Int
    let timestamp: TimeInterval
    let windowStart: TimeInterval
    let windowEnd: TimeInterval
    let confidence: Float
    let rawLabel: String
}
```

Store one session JSON file in the app’s Application Support directory. The UI needs a hole number, shot count, detected label/confidence, event list, and a `Continue` button that advances to the next hole without deleting prior results. Add a simple start/stop camera state and a visible “stationary / processing / waiting for pose” status.

## Stationary detection

Use Core Motion (`CMMotionManager`) to determine whether the phone is steady. Gate expensive Vision/classifier processing until the device has been below configurable accelerometer/rotation thresholds for a short settling period. Keep camera preview available while waiting. Clearly expose this status in the UI and make thresholds easy to tune.

## Validation requirements

Before relying on live video, add a small offline/debug path that loads a saved `[30,17,3]` pose sequence, runs Swift preprocessing, and compares the resulting 19-feature tensor and prediction against Python/reference output. Add tests for:

1. Vision-to-COCO keypoint ordering and coordinate conversion.
2. Invalid-confidence zeroing.
3. Body center/scale fallback behavior.
4. Velocity and rotation-delta calculations.
5. Exact tensor shape/order.
6. Softmax/threshold behavior.
7. Duplicate swing suppression.

Keep the first implementation simple and compilable. Prefer Core ML over ONNX Runtime or PyTorch Mobile. Do not invent a new pose model unless Vision cannot provide the required points. If Core ML generated code uses a slightly different Swift input wrapper, inspect the generated model class and adapt the input construction while preserving the logical shape and names above.
