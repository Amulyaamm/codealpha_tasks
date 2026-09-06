import cv2
import time
from collections import defaultdict, deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolo11n.pt"

# Smaller image size = faster FPS on CPU
IMG_SIZE = 320

CONFIDENCE = 0.35

CAMERA_ID = 0

# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully")


# ============================================================
# INITIALIZE DEEP SORT
# ============================================================

tracker = DeepSort(
    max_age=30,
    n_init=2,
    nms_max_overlap=1.0,
    max_cosine_distance=0.3,
    nn_budget=100,
    embedder="mobilenet",
    half=False
)

print("Deep SORT tracker initialized")


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

# Reduce camera resolution for better FPS
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("✅ Object Detection & Tracking Started")
print("Press Q to quit")
print("Press R to start/stop recording")


# ============================================================
# TRACK HISTORY
# ============================================================

track_history = defaultdict(lambda: deque(maxlen=30))


# ============================================================
# FPS VARIABLES
# ============================================================

prev_time = time.time()

fps = 0


# ============================================================
# RECORDING
# ============================================================

recording = False
video_writer = None


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read webcam frame")
        break

    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    results = model(
        frame,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE,
        verbose=False
    )

    detections = []

    # --------------------------------------------------------
    # EXTRACT DETECTIONS
    # --------------------------------------------------------

    for result in results:

        boxes = result.boxes

        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            class_name = model.names[class_id]

            # Deep SORT format:
            # ([left, top, width, height], confidence, class_name)

            width = x2 - x1
            height = y2 - y1

            detections.append(
                (
                    [x1, y1, width, height],
                    confidence,
                    class_name
                )
            )


    # --------------------------------------------------------
    # DEEP SORT TRACKING
    # --------------------------------------------------------

    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )

def get_color(track_id):
    try:
        track_number = int(track_id)
    except (ValueError, TypeError):
        track_number = abs(hash(str(track_id)))

    return colors[track_number % len(colors)]
    # --------------------------------------------------------
    # COUNTERS
    # --------------------------------------------------------

    person_count = 0
    car_count = 0

    active_tracks = 0


    # --------------------------------------------------------
    # DRAW TRACKS
    # --------------------------------------------------------

    for track in tracks:

        if not track.is_confirmed():
            continue

        if track.time_since_update > 0:
            continue

        active_tracks += 1

        track_id = track.track_id

        ltrb = track.to_ltrb()

        x1, y1, x2, y2 = map(int, ltrb)

        class_name = track.get_det_class()

        if class_name is None:
            class_name = "object"

        # ----------------------------------------------------
        # COUNT OBJECTS
        # ----------------------------------------------------

        if class_name == "person":
            person_count += 1

        if class_name == "car":
            car_count += 1


        # ----------------------------------------------------
        # CENTER POINT
        # ----------------------------------------------------

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)


        # ----------------------------------------------------
        # TRACK HISTORY
        # ----------------------------------------------------

        track_history[track_id].append(
            (center_x, center_y)
        )


        # ----------------------------------------------------
        # DRAW TRACKING TRAIL
        # ----------------------------------------------------

        points = track_history[track_id]

        for i in range(1, len(points)):

            cv2.line(
                frame,
                points[i - 1],
                points[i],
                (255, 255, 0),
                2
            )


        # ----------------------------------------------------
        # DRAW BOUNDING BOX
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label = f"{class_name} | ID {track_id}"

        cv2.rectangle(
            frame,
            (x1, y1 - 30),
            (x1 + 180, y1),
            (0, 255, 0),
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2
        )


        # ----------------------------------------------------
        # CENTER DOT
        # ----------------------------------------------------

        cv2.circle(
            frame,
            (center_x, center_y),
            5,
            (0, 255, 255),
            -1
        )


    # ========================================================
    # FPS CALCULATION
    # ========================================================

    current_time = time.time()

    elapsed = current_time - prev_time

    if elapsed > 0:
        fps = 1 / elapsed

    prev_time = current_time


    # ========================================================
    # PROFESSIONAL DASHBOARD
    # ========================================================

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (230, 145),
        (20, 20, 20),
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.80,
        frame,
        0.20,
        0
    )


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "AI OBJECT TRACKER",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )


    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (15, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # OBJECTS
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Objects: {len(detections)}",
        (15, 77),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # TRACKS
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Tracks: {active_tracks}",
        (15, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # PERSON / CAR COUNT
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Person: {person_count}  Car: {car_count}",
        (15, 127),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1
    )


    # ========================================================
    # TOP RIGHT STATUS
    # ========================================================

    height, width = frame.shape[:2]

    status = "● LIVE"

    cv2.putText(
        frame,
        status,
        (width - 110, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )


    # ========================================================
    # BOTTOM STATUS BAR
    # ========================================================

    cv2.rectangle(
        frame,
        (0, height - 35),
        (width, height),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        "YOLO11 + DEEP SORT",
        (15, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Q: Quit   R: Record",
        (width - 180, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1
    )


    # ========================================================
    # RECORDING
    # ========================================================

    if recording:

        if video_writer is None:

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            video_writer = cv2.VideoWriter(
                "output_tracking.mp4",
                fourcc,
                20,
                (width, height)
            )

        video_writer.write(frame)


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "YOLO + Deep SORT | Object Detection & Tracking",
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


    elif key == ord("r"):

        recording = not recording

        if not recording:

            if video_writer is not None:
                video_writer.release()
                video_writer = None

            print("⏹ Recording stopped")

        else:

            print("🔴 Recording started")


# ============================================================
# CLEANUP
# ============================================================

cap.release()

if video_writer is not None:
    video_writer.release()

cv2.destroyAllWindows()

print("✅ Program stopped")