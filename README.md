# Real-Time Fall Detection

A **real-time fall detection system** using **YOLOv8** for human detection and **Mediapipe Pose** for skeleton-based verification. The system tracks multiple people, detects falls based on bounding box ratios and pose landmarks, and triggers an alert beep when a fall is detected.

## Features
- Real-time human detection using YOLOv8.
- Multi-person tracking with fall verification.
- Fall detection based on bounding box ratio and pose landmarks.
- Highlights fallen individuals with red boxes and labels.
- Visualizes pose landmarks for detailed analysis.
- Alerts with a beep sound when a fall occurs.

## Requirements
- Python 3.10+
- OpenCV, Mediapipe, Torch, Ultralytics, Numpy

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/real-time-fall-detection.git
   cd real-time-fall-detection
