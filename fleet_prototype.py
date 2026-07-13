import cv2
import time
from collections import deque
from ultralytics import YOLO

# 1. Initialize Both AI Models
# Pose model tracks facial landmarks (eyes/nose) for the driver
cabin_model = YOLO("yolov8n-pose.pt")
# Nano model tracks cars, trucks, and traffic lights for the road
road_model = YOLO("yolov8n.pt")

# 2. Setup Camera Capture (0 is your built-in webcam)
cap = cv2.VideoCapture(0)

# Configure video dimensions and frame rates
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 20

# 3. Setup a 5-second Rolling Video Buffer
buffer_length = fps * 5
frame_buffer = deque(maxlen=buffer_length)

# Tracking state variables
distracted_frames = 0
alert_triggered = False

print("🚀 Fleet AI Engine Initialised. Press 'q' to exit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error: Could not read webcam feed.")
        break

    # Save the clean frame into our rolling cloud-upload buffer
    frame_buffer.append(frame.copy())
    
    # Create copies for our two simulated camera views
    cabin_view = frame.copy()
    road_view = frame.copy()

    # ----------------------------------------------------
    # LOOP A: IN-CABIN AI (Driver Monitoring System)
    # ----------------------------------------------------
    cabin_results = cabin_model(cabin_view, verbose=False)
    driver_distracted = False

    for r in cabin_results:
        if r.keypoints is not None and len(r.keypoints.xy) > 0:
            landmarks = r.keypoints.xy[0]  # First detected person
            
            # YOLO Pose indices: 0=Nose, 1=Left Eye, 2=Right Eye
            if len(landmarks) > 2:
                left_eye = landmarks[1]
                right_eye = landmarks[2]
                
                # Simple Logic: If eye keypoints drop to, you turned away or closed them
                if (left_eye[0] == 0 and left_eye[1] == 0) or (right_eye[0] == 0 and right_eye[1] == 0):
                    driver_distracted = True

    # Handle frame thresholds to prevent false positives
    if driver_distracted:
        distracted_frames += 1
    else:
        distracted_frames = max(0, distracted_frames - 1)

    # Trigger Incident Save if distracted for ~1.5 seconds
    if distracted_frames > (fps * 1.5) and not alert_triggered:
        alert_triggered = True
        timestamp = int(time.time())
        filename = f"incident_distraction_{timestamp}.mp4"
        
        # Save the buffer into a clip (Simulating Cloud Save Trigger)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
        while frame_buffer:
            out.write(frame_buffer.popleft())
        out.release()
        
        print(f"🚨 ALERT: Driver Distracted! Cloud clip saved locally as: {filename}")

    # Reset alert latch once driver looks back
    if distracted_frames == 0:
        alert_triggered = False

    # Visual overlay for Cabin Feed
    cabin_annotated = cabin_results[0].plot()
    if distracted_frames > 10:
        cv2.putText(cabin_annotated, "DISTRACTION DETECTED", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # ----------------------------------------------------
    # LOOP B: ROAD-FACING AI (ADAS Tracking)
    # ----------------------------------------------------
    # Filter class indices for roads: 2=car, 3=motorcycle, 5=bus, 7=truck
    road_results = road_model.track(road_view, persist=True, classes=[2, 3, 5, 7], verbose=False)
    road_annotated = road_results[0].plot()

    # ----------------------------------------------------
    # UI DISPLAY: Side-by-Side Dual Monitoring Dashboard
    # ----------------------------------------------------
    # Resize both views so they easily fit next to each other on your screen
    display_w, display_h = 480, 360
    cabin_resize = cv2.resize(cabin_annotated, (display_w, display_h))
    road_resize = cv2.resize(road_annotated, (display_w, display_h))
    
    # Stitches both feeds horizontally into a single dashboard window
    dashboard = cv2.hconcat([cabin_resize, road_resize])
    
    cv2.imshow("Startup Dashboard: Left=Cabin Feed | Right=Road Feed", dashboard)

    # Press 'q' to safely shut down
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
