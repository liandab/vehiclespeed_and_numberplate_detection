import cv2
import numpy as np
import time
import mysql.connector
import os
import torch
import argparse
import requests
from ultralytics import YOLO
from paddleocr import PaddleOCR
from collections import Counter

# 🎯 Load YOLO models
vehicle_model = YOLO('yolov8m.pt').to('cuda') 
number_plate_model = YOLO('best.pt').to('cuda')   

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, required=True, help="Path of the video file")
parser.add_argument("--speed_limit", type=int, default=80, help="Speed limit for violation detection")
parser.add_argument("--real_world_distance", type=float, default=10.0, help="Real-world distance for speed calculation")
args = parser.parse_args()

# 🏎 Load PaddleOCR for Number Plate Recognition
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)

# 🎥 Video source
video_path = args.video_path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

def get_location():
    try:
        res = requests.get('https://ipinfo.io/json')
        data = res.json()
        return data.get("city", "Unknown") + ", " + data.get("region", "Unknown")
    except:
        return "Unknown Location"

LOCATION = get_location()
print("Location:", LOCATION)

# 📡 Connect to MySQL Database
db = mysql.connector.connect(
    host="localhost",
    user="root",  # Apna MySQL username dalna
    password="1234",  # Apna MySQL password dalna
    database="vsnp"
)
cursor = db.cursor()


# ⚡ Get FPS of video
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  

# ✅ Define resolution
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# 🚦 Define speed detection lines & speed limit
LINE1_Y = 500
LINE2_Y = 750
REAL_WORLD_DISTANCE = args.real_world_distance
SPEED_LIMIT = args.speed_limit

# 🏎 Vehicle tracking dictionaries
vehicle_positions = {}  
vehicle_timestamps = {}  
vehicle_speeds = {}  
vehicle_plates = {}  
violations_data = {}  
vehicle_plates_history = {}  # 📌 Stores multiple plate detections for each vehicle
vehicle_id_counter = 1  

# 🎯 YOLO class IDs for vehicles
vehicle_classes = [2, 3, 5, 7]  

# 🔥 Frame Skipping
frame_skip = 2  
frame_count = 0

os.makedirs("violations", exist_ok=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    # 🎯 Run YOLOv8 for vehicle detection
    results = vehicle_model(frame, conf=0.5)
    vehicle_detections = sorted(results[0].boxes, key=lambda b: -b.conf[0].item())[:10]

    for box in vehicle_detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())

        if cls in vehicle_classes and conf > 0.5:
            x_mid = (x1 + x2) // 2
            y_mid = (y1 + y2) // 2

            matched_id = None
            for vid, (prev_x, prev_y) in vehicle_positions.items():
                distance = np.sqrt((x_mid - prev_x) ** 2 + (y_mid - prev_y) ** 2)
                if distance < 70:  
                    matched_id = vid
                    break

            if matched_id is None:
                matched_id = vehicle_id_counter
                vehicle_id_counter += 1

            vehicle_positions[matched_id] = (x_mid, y_mid)

            if (LINE1_Y - 20 <= y2 <= LINE1_Y + 20) and matched_id not in vehicle_timestamps:
                vehicle_timestamps[matched_id] = time.time()

            if (LINE2_Y - 20 <= y2 <= LINE2_Y + 20) and matched_id in vehicle_timestamps and matched_id not in vehicle_speeds:
                time_taken = time.time() - vehicle_timestamps[matched_id]
                if time_taken > 0:
                    speed_kph = (REAL_WORLD_DISTANCE / time_taken) * 3.6
                    vehicle_speeds[matched_id] = int(speed_kph)
                    if matched_id in vehicle_speeds:
                        speed = vehicle_speeds[matched_id]
                        if matched_id in vehicle_plates_history:
                            most_common_plate = Counter(vehicle_plates_history[matched_id]).most_common(1)[0][0]
                            if matched_id not in vehicle_plates or vehicle_plates[matched_id] == "Unknown":
                                vehicle_plates[matched_id] = most_common_plate  # ✅ Store best plate only once
                        number_plate = vehicle_plates.get(matched_id, "Unknown")



                        # 🛑 Insert into Vehicles Table
                        insert_vehicle_query = """
                        INSERT INTO Vehicles (number_plate, speed_kph, location)
                        VALUES (%s, %s, %s)
                        """
                        cursor.execute(insert_vehicle_query, (number_plate, speed, LOCATION))
                        db.commit()
  
                        def send_vehicle_update(vehicle_data):
                            try:
                                requests.post('http://localhost:5000/update_vehicle', json=vehicle_data)
                            except Exception as e:
                                print("Error sending update:", e)
                        



            if matched_id in vehicle_speeds and vehicle_speeds[matched_id] > SPEED_LIMIT and matched_id not in violations_data:
                snapshot = frame[y1:y2, x1:x2]
                snapshot_path = f"violations/vehicle_{matched_id}.jpg"
                cv2.imwrite(snapshot_path, snapshot)  

                violations_data[matched_id] = {
                    "Speed": vehicle_speeds[matched_id],
                    "Number Plate": "Pending OCR...",
                    "Timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                    "Snapshot": snapshot_path
                }
                
                if matched_id in vehicle_speeds and vehicle_speeds[matched_id] > SPEED_LIMIT:
                    speed = vehicle_speeds[matched_id]

                    # 🛑 Insert into Violations Table
                    insert_violation_query = """
                    INSERT INTO Violations (number_plate, speed_kph, speed_limit, location, snapshot)
                    VALUES (%s, %s, %s, %s, %s)
                    """
                    cursor.execute(insert_violation_query, (number_plate, speed, SPEED_LIMIT, LOCATION, snapshot_path))

                    db.commit()

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame, f'ID: {matched_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            if matched_id in vehicle_speeds:
                cv2.putText(frame, f"{vehicle_speeds[matched_id]} Km/h", (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # **Number Plate Detection**
    plate_results = number_plate_model(frame, conf=0.6)
    for plate in plate_results[0].boxes:
        px1, py1, px2, py2 = map(int, plate.xyxy[0])

        plate_crop = frame[py1:py2, px1:px2]
        plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        plate_gray = cv2.resize(plate_gray, (200, 50))

        ocr_result = ocr.ocr(plate_gray, cls=True)
        plate_text = "N/A" if not ocr_result or not ocr_result[0] else ocr_result[0][0][1][0]

        closest_vehicle = None
        min_distance = float('inf')
        plate_mid_x = (px1 + px2) // 2
        plate_mid_y = (py1 + py2) // 2

        for vid, (vx, vy) in vehicle_positions.items():
            distance = np.sqrt((plate_mid_x - vx) ** 2 + (plate_mid_y - vy) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_vehicle = vid

        if closest_vehicle:
            if closest_vehicle not in vehicle_plates_history:
                vehicle_plates_history[closest_vehicle] = []
            vehicle_plates_history[closest_vehicle].append(plate_text)

        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
        cv2.putText(frame, plate_text, (px1, py1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    for vid in violations_data:
        if vid in vehicle_plates_history:
            most_common_plate = Counter(vehicle_plates_history[vid]).most_common(1)[0][0]
            violations_data[vid]["Number Plate"] = most_common_plate

        # 🚦 Draw Speed Limit Lines
    cv2.line(frame, (0, LINE1_Y), (FRAME_WIDTH, LINE1_Y), (0, 0, 255), 3)  # Red Line
    cv2.line(frame, (0, LINE2_Y), (FRAME_WIDTH, LINE2_Y), (255, 0, 0), 3)  # Blue Line


    cv2.imshow('Vehicle Speed & Number Plate Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n🚗🚦 **Speed Violations Report** 🚦🚗\n")
for vid, details in violations_data.items():
    print(f"ID {vid}: {details}")


print("CUDA Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU Found")


cursor.close()
db.close()

