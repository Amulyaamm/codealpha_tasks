import cv2
import time
import math
from collections import Counter

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "yolo11n.pt"

CAMERA_INDEX = 0

CONFIDENCE_THRESHOLD = 0.35

# CPU-friendly image size
IMG_SIZE = 640

# Process every frame
FRAME_SKIP = 1

# Classes we want to track
TARGET_CLASSES = {
    "person",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "cell phone",
}

# ============================================================
# COLORS - BGR FORMAT
# ============================================================

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
BLUE = (255, 100, 0)
GRAY = (80, 80, 80)
DARK = (20, 20, 20)

TRACK_COLORS = [
    (255, 100, 100),
    (100, 255, 100),
    (100, 200, 255),
    (255, 180, 80),
    (180, 100, 255),
    (80, 255, 220),
    (255, 120, 200),
    (150, 255, 150),
]


# ============================================================
# SAFE TRACK COLOR FUNCTION
# ============================================================

def get_color(track_id):
    """
    Deep SORT track IDs may be integers or strings.
    Convert safely before using modulo.
    """

    try:
        track_number = int(track_id)
    except (ValueError, TypeError):
        track_number = abs(hash(str(track_id)))

    return TRACK_COLORS[track_number % len(TRACK_COLORS)]


# ============================================================
# DRAW ROUNDED RECTANGLE
# ============================================================

def rounded_rectangle(image, pt1, pt2, color, thickness=1, radius=10):
    x1, y1 = pt1
    x2, y2 = pt2

    cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness)

    cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

    cv2.ellipse(
        image,
        (x1 + radius, y1 + radius),
        (radius, radius),
        180,
        0,
        90,
        color,
        thickness
    )

    cv2.ellipse(
        image,
        (x2 - radius, y1 + radius),
        (radius, radius),
        270,
        0,
        90,
        color,
        thickness
    )

    cv2.ellipse(
        image,
        (x1 + radius, y2 - radius),
        (radius, radius),
        90,
        0,
        90,
        color,
        thickness
    )

    cv2.ellipse(
        image,
        (x2 - radius, y2 - radius),
        (radius, radius),
        0,
        0,
        90,
        color,
        thickness
    )


# ============================================================
# DRAW CORNER BOX
# ============================================================

def draw_corner_box(frame, x1, y1, x2, y2, color, thickness=2, corner=14):

    # Top-left
    cv2.line(frame, (x1, y1), (x1 + corner, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner), color, thickness)

    # Top-right
    cv2.line(frame, (x2, y1), (x2 - corner, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner), color, thickness)

    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + corner, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner), color, thickness)

    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - corner, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner), color, thickness)


# ============================================================
# DRAW TEXT WITH SHADOW
# ============================================================

def put_text(frame, text, position, scale=0.6, color=WHITE, thickness=1):

    x, y = position

    cv2.putText(
        frame,
        text,
        (x + 2, y + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        BLACK,
        thickness + 2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# DRAW HEADER
# ============================================================

def draw_header(frame, fps, object_count, tracked_count):

    height, width = frame.shape[:2]

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (width, 72),
        (10, 10, 10),
        -1
    )

    frame[:] = cv2.addWeighted(overlay, 0.82, frame, 0.18, 0)

    # Main title
    put_text(
        frame,
        "VISIONTRACK AI",
        (24, 30),
        0.8,
        CYAN,
        2
    )

    put_text(
        frame,
        "YOLO11 + DEEP SORT",
        (26, 54),
        0.42,
        WHITE,
        1
    )

    # FPS
    put_text(
        frame,
        f"FPS  {fps:05.1f}",
        (width - 300, 29),
        0.55,
        GREEN,
        2
    )

    # Detection count
    put_text(
        frame,
        f"OBJECTS  {object_count:02d}",
        (width - 170, 29),
        0.5,
        YELLOW,
        1
    )

    # Tracking status
    status_color = GREEN if tracked_count > 0 else RED

    cv2.circle(
        frame,
        (width - 285, 53),
        5,
        status_color,
        -1
    )

    put_text(
        frame,
        "TRACKING ACTIVE" if tracked_count > 0 else "SEARCHING",
        (width - 270, 58),
        0.4,
        status_color,
        1
    )


# ============================================================
# DRAW CROSSHAIR
# ============================================================

def draw_crosshair(frame):

    height, width = frame.shape[:2]

    cx = width // 2
    cy = height // 2

    size = 20

    cv2.line(
        frame,
        (cx - size, cy),
        (cx - 5, cy),
        CYAN,
        1
    )

    cv2.line(
        frame,
        (cx + 5, cy),
        (cx + size, cy),
        CYAN,
        1
    )

    cv2.line(
        frame,
        (cx, cy - size),
        (cx, cy - 5),
        CYAN,
        1
    )

    cv2.line(
        frame,
        (cx, cy + 5),
        (cx, cy + size),
        CYAN,
        1
    )

    cv2.circle(
        frame,
        (cx, cy),
        2,
        CYAN,
        -1
    )


# ============================================================
# DRAW FOOTER
# ============================================================

def draw_footer(frame):

    height, width = frame.shape[:2]

    y = height - 42

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, y),
        (width, height),
        (10, 10, 10),
        -1
    )

    frame[:] = cv2.addWeighted(
        overlay,
        0.80,
        frame,
        0.20,
        0
    )

    put_text(
        frame,
        "Q  QUIT",
        (20, height - 15),
        0.42,
        WHITE,
        1
    )

    put_text(
        frame,
        "S  SAVE SCREENSHOT",
        (110, height - 15),
        0.42,
        WHITE,
        1
    )

    put_text(
        frame,
        "REAL-TIME OBJECT DETECTION & TRACKING",
        (width - 360, height - 15),
        0.38,
        GRAY,
        1
    )


# ============================================================
# DRAW OBJECT STATISTICS
# ============================================================

def draw_statistics(frame, counts):

    x = 20
    y = 105

    put_text(
        frame,
        "DETECTION MONITOR",
        (x, y),
        0.45,
        CYAN,
        1
    )

    y += 25

    if not counts:

        put_text(
            frame,
            "No objects detected",
            (x, y),
            0.4,
            GRAY,
            1
        )

        return

    for class_name, count in counts.items():

        label = f"{class_name.upper():<14} {count:02d}"

        put_text(
            frame,
            label,
            (x, y),
            0.38,
            WHITE,
            1
        )

        y += 20

        if y > frame.shape[0] - 70:
            break


# ============================================================
# START YOLO
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("YOLO model loaded")


# ============================================================
# START DEEP SORT
# ============================================================

tracker = DeepSort(
    max_age=30,
    n_init=2,
    max_iou_distance=0.7,
    max_cosine_distance=0.2,
    nn_budget=100
)


# ============================================================
# START WEBCAM
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():

    print("❌ Could not open webcam.")

    raise SystemExit


# Camera resolution

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

# Try to reduce camera buffering

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


print("✅ Object Detection & Tracking Started")
print("Press Q to quit")
print("Press S to save screenshot")


# ============================================================
# FPS VARIABLES
# ============================================================

prev_time = time.time()

fps = 0.0

frame_number = 0

screenshot_number = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("❌ Failed to read webcam frame")

        break


    frame_number += 1


    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    current_time = time.time()

    elapsed = current_time - prev_time

    if elapsed > 0:

        fps = 1.0 / elapsed

    prev_time = current_time


    # --------------------------------------------------------
    # Resize for faster CPU inference
    # --------------------------------------------------------

    display_frame = frame.copy()


    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    results = model.predict(
        display_frame,
        imgsz=IMG_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
        device="cpu"
    )


    # --------------------------------------------------------
    # PREPARE DEEP SORT DETECTIONS
    # --------------------------------------------------------

    detections = []

    detected_names = []


    result = results[0]


    if result.boxes is not None:

        for box in result.boxes:

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            class_name = model.names[class_id]


            # Only track selected objects

            if class_name not in TARGET_CLASSES:

                continue


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )


            width = x2 - x1
            height = y2 - y1


            if width <= 0 or height <= 0:

                continue


            # Deep SORT format:
            # ([left, top, width, height], confidence, class)

            detections.append(
                (
                    [x1, y1, width, height],
                    confidence,
                    class_name
                )
            )


            detected_names.append(class_name)


    # --------------------------------------------------------
    # DEEP SORT TRACKING
    # --------------------------------------------------------

    tracks = tracker.update_tracks(
        detections,
        frame=display_frame
    )


    # --------------------------------------------------------
    # DRAW TRACKS
    # --------------------------------------------------------

    active_tracks = 0

    track_counts = Counter()


    for track in tracks:

        if not track.is_confirmed():

            continue


        if track.time_since_update > 1:

            continue


        active_tracks += 1


        track_id = track.track_id


        # Get bounding box

        try:

            ltrb = track.to_ltrb()

            x1, y1, x2, y2 = map(
                int,
                ltrb
            )

        except Exception:

            continue


        # Class name

        class_name = track.get_det_class()

        if class_name is None:

            class_name = "object"


        track_counts[class_name] += 1


        # Color

        color = get_color(track_id)


        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        draw_corner_box(
            display_frame,
            x1,
            y1,
            x2,
            y2,
            color,
            thickness=2,
            corner=15
        )


        # ----------------------------------------------------
        # Center point
        # ----------------------------------------------------

        center_x = int((x1 + x2) / 2)

        center_y = int((y1 + y2) / 2)


        cv2.circle(
            display_frame,
            (center_x, center_y),
            4,
            color,
            -1
        )


        # ----------------------------------------------------
        # Tracking line
        # ----------------------------------------------------

        cv2.line(
            display_frame,
            (center_x, center_y),
            (center_x, y2),
            color,
            1
        )


        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = f"{class_name.upper()}  ID:{track_id}"


        (text_width, text_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            1
        )


        label_x = x1

        label_y = max(
            y1 - 8,
            text_height + 10
        )


        # Label background

        cv2.rectangle(
            display_frame,
            (
                label_x,
                label_y - text_height - 8
            ),
            (
                label_x + text_width + 10,
                label_y + 3
            ),
            color,
            -1
        )


        # Label text

        cv2.putText(
            display_frame,
            label,
            (
                label_x + 5,
                label_y - 2
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            BLACK,
            1,
            cv2.LINE_AA
        )


    # --------------------------------------------------------
    # OBJECT COUNTS
    # --------------------------------------------------------

    object_counts = Counter(detected_names)


    total_objects = sum(
        object_counts.values()
    )


    # --------------------------------------------------------
    # DRAW UI
    # --------------------------------------------------------

    draw_header(
        display_frame,
        fps,
        total_objects,
        active_tracks
    )


    draw_statistics(
        display_frame,
        object_counts
    )


    draw_crosshair(
        display_frame
    )


    draw_footer(
        display_frame
    )


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    cv2.imshow(
        "VISIONTRACK AI - Object Detection & Tracking",
        display_frame
    )


    # --------------------------------------------------------
    # KEYBOARD CONTROLS
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF


    # Quit

    if key == ord("q") or key == ord("Q"):

        break


    # Screenshot

    elif key == ord("s") or key == ord("S"):

        screenshot_number += 1

        filename = (
            f"visiontrack_screenshot_"
            f"{screenshot_number}.jpg"
        )

        cv2.imwrite(
            filename,
            display_frame
        )

        print(
            f"📸 Screenshot saved: {filename}"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("✅ Object Detection & Tracking stopped.")