from ultralytics import YOLO
import cv2
import math
import os
import csv
from collections import defaultdict, deque
import easyocr
import re
from collections import Counter

# =====================================
# MODELS
# =====================================

smoke_model = YOLO("best.pt")

vehicle_model = YOLO("yolov8n.pt")

plate_model = YOLO("plate.pt")
reader = easyocr.Reader(['en'])

# =====================================
# INPUT VIDEO
# =====================================



# =====================================
# OUTPUT FOLDERS
# =====================================

evidence_dir = "evidence"
os.makedirs(evidence_dir, exist_ok=True)

# =====================================
# CSV LOG
# =====================================

log_file = os.path.join(evidence_dir, "evidence_log.csv")

with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Vehicle_ID",
        "Vehicle_Type",
        "Timestamp",
        "Smoke_Count",
        "Frame_Number",
        "Plate_Number"
    ])

# =====================================
# TRACKER DATA
# =====================================

stable_tracks = {}
next_stable_id = 1
used_ids_this_frame = set()
track_age = defaultdict(int)
smoke_history = defaultdict(lambda: deque(maxlen=150))
saved_suspects = set()
vehicle_ocr_history = defaultdict(list)

# =====================================
# BEST FRAME STORAGE
# =====================================

best_vehicle_crop = {}
best_vehicle_frame = {}
best_vehicle_area = {}

# =====================================
# VEHICLE CLASSES
# =====================================

vehicle_classes = [
    2,  # car
    3,  # motorcycle
    5,  # bus
    7   # truck
]

# =====================================
# HELPER FUNCTIONS
# =====================================
def send_event(
    stable_id,
    plate_number,
    smoke_count,
    time_string
):

    payload = {
        "vehicle_id": stable_id,
        "plate": plate_number,
        "smoke_count": smoke_count,
        "timestamp": time_string,
        "event_type": "SMOKE_VIOLATION"
    }

    print("\nEVENT GENERATED")
    print(payload)
def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = box_area((ix1, iy1, ix2, iy2))
    union = box_area(boxA) + box_area(boxB) - inter

    if union == 0:
        return 0

    return inter / union

# =====================================
# STABLE TRACKER
# =====================================

def match_or_create_stable_id(vehicle_box, vehicle_name, frame_no):
    global next_stable_id

    vc = center(vehicle_box)
    best_id = None
    best_score = -1

    for stable_id, data in stable_tracks.items():
        if stable_id in used_ids_this_frame:
            continue

        frames_missing = frame_no - data["last_seen"]
        if frames_missing > 300:
            continue

        old_box = data["box"]
        overlap = iou(vehicle_box, old_box)

        if overlap < 0.01:
            continue

        d = distance(vc, center(old_box))
        size_ratio = box_area(vehicle_box) / max(box_area(old_box), 1)

        if not (0.3 <= size_ratio <= 3.0):
            continue

        score = overlap * 5
        score += max(0, (150 - d) / 150)

        if vehicle_name == data["name"]:
            score += 1

        if score > best_score:
            best_score = score
            best_id = stable_id

    if best_id is not None:
        stable_tracks[best_id]["box"] = vehicle_box
        stable_tracks[best_id]["last_seen"] = frame_no
        stable_tracks[best_id]["name"] = vehicle_name
        used_ids_this_frame.add(best_id)
        return best_id

    stable_id = next_stable_id
    next_stable_id += 1

    stable_tracks[stable_id] = {
        "box": vehicle_box,
        "name": vehicle_name,
        "last_seen": frame_no
    }

    used_ids_this_frame.add(stable_id)
    return stable_id

# =====================================
# SMOKE ASSOCIATION
# =====================================

def smoke_near_vehicle(smoke_box, vehicle_box):
    sx1, sy1, sx2, sy2 = smoke_box
    vx1, vy1, vx2, vy2 = vehicle_box

    vw = vx2 - vx1
    vh = vy2 - vy1

    expanded = (
        vx1 - int(0.5 * vw),
        vy1 - int(0.5 * vh),
        vx2 + int(1.0 * vw),
        vy2 + int(1.0 * vh)
    )

    ex1, ey1, ex2, ey2 = expanded
    sc = center(smoke_box)

    return (ex1 <= sc[0] <= ex2 and ey1 <= sc[1] <= ey2)


# =====================================
# PLATE OCR HELPERS
# =====================================

def fix_plate_ocr(plate_number):
    """
    Fix common OCR misreads for Indian plates: XX00XX0000
    Position 0,1 = letters
    Position 2,3 = digits
    Position 4,5 = letters
    Position 6-9  = digits
    """
    if len(plate_number) != 10:
        return plate_number

    result = list(plate_number)

    digit_positions = [2, 3, 6, 7, 8, 9]
    letter_positions = [0, 1, 4, 5]

    letter_fixes = {'0': 'O', '1': 'I', '8': 'B', '5': 'S'}
    digit_fixes  = {'O': '0', 'I': '1', 'B': '8', 'S': '5', 'Z': '2', 'G': '6'}

    for i in digit_positions:
        if result[i] in digit_fixes:
            result[i] = digit_fixes[result[i]]

    for i in letter_positions:
        if result[i] in letter_fixes:
            result[i] = letter_fixes[result[i]]

    return "".join(result)


def read_plate_from_crop(plate_crop, stable_id):
    """
    Run OCR on plate crop and return best valid Indian plate string or None.
    """
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    # Resize to fixed Indian plate ratio
    gray = cv2.resize(gray, (333, 75))
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    results = reader.readtext(
        thresh,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        detail=1,
        paragraph=False
    )

    print(f"  Raw OCR for Vehicle {stable_id}: {results}")

    # Sort by confidence high to low
    results_sorted = sorted(results, key=lambda x: x[2], reverse=True)

    # Step 1: Try each chunk individually
    for ocr_item in results_sorted:
        text = re.sub(r'[^A-Z0-9]', '', ocr_item[1].upper())
        fixed = fix_plate_ocr(text)
        print(f"  Chunk: '{text}' -> Fixed: '{fixed}' (conf: {ocr_item[2]:.2f})")
        if re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$', fixed):
            print(f"  VALID (single chunk): {fixed}")
            return fixed

    # Step 2: Join top chunks and use sliding window of 10
    joined = "".join([
        re.sub(r'[^A-Z0-9]', '', r[1].upper())
        for r in results_sorted[:6]
    ])
    print(f"  Joined: '{joined}'")

    for i in range(len(joined) - 9):
        chunk = joined[i:i+10]
        fixed = fix_plate_ocr(chunk)
        if re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$', fixed):
            print(f"  VALID (sliding window): {fixed}")
            return fixed

    # Step 3: Return best partial (8-11 chars) as fallback
    for ocr_item in results_sorted:
        text = re.sub(r'[^A-Z0-9]', '', ocr_item[1].upper())
        if 8 <= len(text) <= 11:
            fixed = fix_plate_ocr(text)
            print(f"  Partial fallback: {fixed}")
            return fixed

    print(f"  No plate found")
    return None


# =====================================
# VIDEO INPUT
# =====================================

USE_CAMERA = False

video_path = r"C:\Users\E028.26\Downloads\WhatsApp Video 2026-06-21 at 12.31.03 PM.mp4"

if USE_CAMERA:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

print("Video opened successfully")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# =====================================
# OUTPUT VIDEO
# =====================================

out = cv2.VideoWriter(
    "stable_suspect_output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

frame_no = 0
recording = False
video_writer = None
current_vehicles = []

# =====================================
# MAIN LOOP
# =====================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_no += 1

    if frame_no % 30 == 0:
        print("Processing frame:", frame_no)

    used_ids_this_frame.clear()

    # ----------------------
    # Adaptive Processing
    # ----------------------

    processing_interval = 1

    if frame_no % processing_interval != 0:
        out.write(frame)
        continue

    # ----------------------
    # Smoke Detection
    # ----------------------

    smoke_results = smoke_model(
        frame,
        conf=0.18,
        imgsz=960,
        verbose=False
    )

    smoke_boxes = []

    if len(smoke_results) > 0:
        for box in smoke_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            smoke_boxes.append((x1, y1, x2, y2))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame, "Smoke", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

    # ----------------------
    # Vehicle Detection
    # ----------------------

    vehicle_results = vehicle_model(
        frame,
        conf=0.20,
        imgsz=960,
        verbose=False
    )

    current_vehicles = []

    for box in vehicle_results[0].boxes:
        cls = int(box.cls[0])

        if cls not in vehicle_classes:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        vehicle_box = (x1, y1, x2, y2)
        name = vehicle_model.names[cls]

        stable_id = match_or_create_stable_id(vehicle_box, name, frame_no)
        track_age[stable_id] += 1
        current_vehicles.append((stable_id, vehicle_box, name))

    # ----------------------
    # PROCESS VEHICLES
    # ----------------------

    for stable_id, vehicle_box, name in current_vehicles:
        smoke_found = False

        for smoke_box in smoke_boxes:
            if smoke_near_vehicle(smoke_box, vehicle_box):
                smoke_found = True
                break

        smoke_history[stable_id].append(1 if smoke_found else 0)
        smoke_count = sum(smoke_history[stable_id])
        checked = len(smoke_history[stable_id])

        x1, y1, x2, y2 = vehicle_box
        crop = frame[y1:y2, x1:x2]
        area = (x2 - x1) * (y2 - y1)

        # ------------------
        # SAVE BEST FRAME
        # ------------------

        if crop.size > 0:
            if stable_id not in best_vehicle_area:
                best_vehicle_area[stable_id] = area
                best_vehicle_crop[stable_id] = crop.copy()
                best_vehicle_frame[stable_id] = frame.copy()
            elif area > best_vehicle_area[stable_id]:
                best_vehicle_area[stable_id] = area
                best_vehicle_crop[stable_id] = crop.copy()
                best_vehicle_frame[stable_id] = frame.copy()

        suspect = (
            checked >= 150
            and smoke_count >= 60
            and track_age[stable_id] >= 100
        )

        if suspect:
            color = (0, 0, 255)
            label = f"SUSPECT Vehicle-{stable_id}"

            if not recording:
                proof_path = os.path.join(
                    evidence_dir,
                    f"Vehicle_{stable_id}_proof.mp4"
                )

                video_writer = cv2.VideoWriter(
                    proof_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height)
                )

                recording = True

            if stable_id not in saved_suspects:
                timestamp_sec = frame_no / fps
                minutes = int(timestamp_sec // 60)
                seconds = int(timestamp_sec % 60)
                time_string = f"{minutes:02d}:{seconds:02d}"

                frame_path = os.path.join(
                    evidence_dir, f"Vehicle_{stable_id}_frame.jpg"
                )
                crop_path = os.path.join(
                    evidence_dir, f"Vehicle_{stable_id}_crop.jpg"
                )

                if stable_id in best_vehicle_frame:
                    cv2.imwrite(frame_path, best_vehicle_frame[stable_id])

                if stable_id in best_vehicle_crop:
                    cv2.imwrite(crop_path, best_vehicle_crop[stable_id])

                # ------------------
                # PLATE DETECTION
                # ------------------

                vehicle_crop = best_vehicle_crop.get(stable_id, None)
                plate_results = None

                if vehicle_crop is not None and vehicle_crop.size > 0:
                    try:
                        plate_results = plate_model(
                            vehicle_crop, conf=0.10, verbose=False
                        )
                    except Exception as e:
                        print(f"Plate model error for Vehicle {stable_id}: {e}")

                if plate_results is not None:
                    for plate_box in plate_results[0].boxes:
                        px1, py1, px2, py2 = map(int, plate_box.xyxy[0])

                        # Small padding
                        pad = 5
                        px1 = max(0, px1 - pad)
                        py1 = max(0, py1 - pad)
                        px2 = min(vehicle_crop.shape[1], px2 + pad)
                        py2 = min(vehicle_crop.shape[0], py2 + pad)

                        plate_crop = vehicle_crop[py1:py2, px1:px2]

                        if plate_crop.size == 0:
                            continue

                        # Save plate image
                        plate_path = os.path.join(
                            evidence_dir, f"Vehicle_{stable_id}_plate.jpg"
                        )
                        cv2.imwrite(plate_path, plate_crop)

                        # Run OCR
                        plate_number = read_plate_from_crop(plate_crop, stable_id)

                        if plate_number:
                            vehicle_ocr_history[stable_id].append(plate_number)

                # ------------------
                # FINAL PLATE
                # ------------------

                if vehicle_ocr_history[stable_id]:
                    final_plate = Counter(
                        vehicle_ocr_history[stable_id]
                    ).most_common(1)[0][0]
                else:
                    final_plate = "UNKNOWN"

                with open(log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        stable_id,
                        name,
                        time_string,
                        smoke_count,
                        frame_no,
                        final_plate
                    ])

                send_event(
                    stable_id,
                    final_plate,
                    smoke_count,
                    time_string
                )

                print(f"Suspect {stable_id} saved | Plate: {final_plate}")

                saved_suspects.add(stable_id)

        else:
            color = (255, 0, 0)
            label = f"Vehicle-{stable_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    out.write(frame)

    if recording and video_writer is not None:
        video_writer.write(frame)

    cv2.imshow(
        "Smoke Emission Monitor",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# =====================================
# CLEANUP
# =====================================

cap.release()
out.release()

if video_writer is not None:
    video_writer.release()

cv2.destroyAllWindows()
print()
print("Finished Processing")
print("Total suspects:", len(saved_suspects))
print("Output Video:", "stable_suspect_output.mp4")