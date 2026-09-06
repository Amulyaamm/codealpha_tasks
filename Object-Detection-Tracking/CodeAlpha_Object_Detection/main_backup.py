import cv2
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO("yolo11n.pt")

# -----------------------------
# Initialize Deep SORT
# -----------------------------
tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_iou_distance=0.7
)

# -----------------------------
# Open webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("====================================")
print(" YOLO + DEEP SORT OBJECT TRACKING")
print("====================================")
print("✅ Webcam started")
print("Press Q to quit")

# FPS
prev_time = 0

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read frame")
        break

    # -----------------------------
    # YOLO object detection
    # -----------------------------
    results = model(
        frame,
        conf=0.5,
        verbose=False
    )

    detections = []

    for result in results:

        boxes = result.boxes

        for box in boxes:

            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Confidence
            confidence = float(box.conf[0])

            # Class ID
            class_id = int(box.cls[0])

            # Class name
            class_name = model.names[class_id]

            # Convert to Deep SORT format
            width = x2 - x1
            height = y2 - y1

            detections.append(
                (
                    [x1, y1, width, height],
                    confidence,
                    class_name
                )
            )

    # -----------------------------
    # Deep SORT tracking
    # -----------------------------
    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )

    # Count objects
    object_count = 0

    # -----------------------------
    # Draw tracking results
    # -----------------------------
    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        ltrb = track.to_ltrb()

        x1, y1, x2, y2 = map(int, ltrb)

        # Get class name
        class_name = track.get_det_class()

        if class_name is None:
            class_name = "object"

        object_count += 1

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label
        label = f"{class_name} ID:{track_id}"

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # -----------------------------
    # Calculate FPS
    # -----------------------------
    current_time = time.time()

    if prev_time != 0:
        fps = 1 / (current_time - prev_time)
    else:
        fps = 0

    prev_time = current_time

    # -----------------------------
    # Display information
    # -----------------------------
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Objects: {object_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Title
    cv2.putText(
        frame,
        "YOLO + Deep SORT",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # -----------------------------
    # Show output
    # -----------------------------
    cv2.imshow(
        "CodeAlpha - Object Detection and Tracking",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# Release resources
# -----------------------------
cap.release()
cv2.destroyAllWindows()

print("✅ Program stopped")