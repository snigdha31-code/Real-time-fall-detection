import cv2
from ultralytics import YOLO
import winsound
import mediapipe as mp
import math
import csv
import os
from datetime import datetime


model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

FALL_RATIO_THRESHOLD = 0.8
FALL_FRAMES_REQUIRED = 3  # consecutive frames to confirm fall


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils


person_tracker = {}  # track each person
frame_count = 0


csv_file = "fall_analytics.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "PersonID", "FallDetected", "NumPeople", "FallsPerPerson"])

analytics = {'num_people': 0}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    current_person_ids = []

    # Loop through YOLO detections
    for idx, (box, cls) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.cls)):
        x1, y1, x2, y2 = map(int, box)
        width = x2 - x1
        height = y2 - y1
        ratio = height / width if width != 0 else 0

        # Only humans
        if int(cls) != 0:
            continue

        person_id = idx
        current_person_ids.append(person_id)

        if person_id not in person_tracker:
            person_tracker[person_id] = {'fall_count': 0, 'fallen': False, 'total_falls': 0}

        # Check ratio for fall
        if ratio < FALL_RATIO_THRESHOLD:
            person_tracker[person_id]['fall_count'] += 1
        else:
            person_tracker[person_id]['fall_count'] = 0
            person_tracker[person_id]['fallen'] = False

        if (person_tracker[person_id]['fall_count'] >= FALL_FRAMES_REQUIRED
            and not person_tracker[person_id]['fallen']):
            winsound.Beep(1000, 500)
            person_tracker[person_id]['fallen'] = True
            person_tracker[person_id]['total_falls'] += 1

            # Save to CSV
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), person_id, True, len(set(current_person_ids)), person_tracker[person_id]['total_falls']])

     
        color = (0, 255, 0)
        label = "Person"
        if person_tracker[person_id]['fallen']:
            color = (0, 0, 255)
            label = "FALL DETECTED!"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

   
    for idx, (box, cls) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.cls)):
        x1, y1, x2, y2 = map(int, box)
        width = x2 - x1
        height = y2 - y1
        ratio = height / width if width != 0 else 0

        # Only humans
        if int(cls) != 0:
            continue

        person_id = idx
        current_person_ids.append(person_id)

        if person_id not in person_tracker:
            person_tracker[person_id] = {'fall_count': 0, 'fallen': False, 'total_falls': 0}

       
        color = (0, 255, 0)
        label = "Person"
        if person_tracker[person_id]['fallen']:
            color = (0, 0, 255)
            label = "FALL DETECTED!"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue
        person_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(person_rgb)

        if pose_results.pose_landmarks:
            # Adjust landmarks to original frame coordinates
            for lm in pose_results.pose_landmarks.landmark:
                cx = int(lm.x * width) + x1
                cy = int(lm.y * height) + y1
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                # Optionally draw connections if needed
                # mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_draw.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2))

    num_people = len(set(current_person_ids))
    if num_people != analytics['num_people']:
        analytics['num_people'] = num_people
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'Analytics', False, num_people, ''])

  
    lost_ids = set(person_tracker.keys()) - set(current_person_ids)
    for lost_id in lost_ids:
        del person_tracker[lost_id]

    cv2.imshow("Fall Detection + Skeleton", frame)
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
