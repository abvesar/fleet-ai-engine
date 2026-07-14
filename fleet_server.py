import cv2
import time
import math
import atexit
import threading
from collections import deque
from flask import Flask, Response
import mediapipe as mp

# 1. Initialize Flask Web Server
app = Flask(__name__)

# 2. Global stream resources created lazily so the module can be imported safely.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = None
cap = None
frame_width = 640
frame_height = 480
fps = 20
frame_buffer = deque(maxlen=fps * 5)
latest_frame_bytes = None
capture_thread = None
capture_running = False
stream_lock = threading.Lock()
frame_ready = threading.Condition(stream_lock)


def initialize_stream_resources():
    """Open the camera and create the face mesh detector once."""
    global cap, face_mesh, frame_width, frame_height, fps, frame_buffer, buffer_length, DROWSY_FRAMES_LIMIT, drowsy_counter, alert_active, capture_thread, capture_running

    if cap is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            cap = None
            raise RuntimeError("Unable to open camera 0")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps_value = cap.get(cv2.CAP_PROP_FPS)
        fps = int(fps_value) if fps_value > 0 else 20

        buffer_length = fps * 5
        DROWSY_FRAMES_LIMIT = int(fps * 1.5)
        frame_buffer = deque(maxlen=buffer_length)
        drowsy_counter = 0
        alert_active = False

    if face_mesh is None:
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    if not capture_running:
        capture_running = True
        capture_thread = threading.Thread(target=capture_loop, daemon=True)
        capture_thread.start()


def cleanup_stream_resources():
    """Release camera and MediaPipe resources safely."""
    global cap, face_mesh, capture_running, capture_thread

    capture_running = False
    with frame_ready:
        frame_ready.notify_all()

    if capture_thread is not None and capture_thread.is_alive() and threading.current_thread() is not capture_thread:
        capture_thread.join(timeout=2)
    capture_thread = None

    if face_mesh is not None:
        face_mesh.close()
        face_mesh = None

    if cap is not None:
        cap.release()
        cap = None


atexit.register(cleanup_stream_resources)

# Metrics for calculating Eye Aspect Ratio (EAR) to detect closed eyes
# Left eye landmark indices inside MediaPipe Face Mesh
LEFT_EYE_TOP_BOTTOM = [386, 374]
LEFT_EYE_LEFT_RIGHT = [362, 263]
# Right eye landmark indices
RIGHT_EYE_TOP_BOTTOM = [159, 145]
RIGHT_EYE_LEFT_RIGHT = [33, 133]

EAR_THRESHOLD = 0.20  # If eye openness falls below this, the eye is closed
DROWSY_FRAMES_LIMIT = int(fps * 1.5)  # 1.5 seconds threshold
drowsy_counter = 0
alert_active = False

def calculate_ear(landmarks, top_bottom_idx, left_right_idx):
    """Calculates the Eye Aspect Ratio (EAR) to measure eye openness."""
    # Vertical distance
    p_top = landmarks[top_bottom_idx[0]]
    p_bottom = landmarks[top_bottom_idx[1]]
    dist_v = math.sqrt((p_top.x - p_bottom.x)**2 + (p_top.y - p_bottom.y)**2)
    
    # Horizontal distance
    p_left = landmarks[left_right_idx[0]]
    p_right = landmarks[left_right_idx[1]]
    dist_h = math.sqrt((p_left.x - p_right.x)**2 + (p_left.y - p_right.y)**2)
    
    if dist_h == 0:
        return 0
    return dist_v / dist_h


def capture_loop():
    """Read frames, run detection, and publish the latest encoded JPEG for all clients."""
    global drowsy_counter, alert_active, latest_frame_bytes

    while capture_running and cap is not None and cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Store clean frame in the rolling incident buffer.
        frame_buffer.append(frame.copy())

        # Convert BGR image to RGB for MediaPipe processing.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        is_drowsy_this_frame = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                # Calculate EAR for both eyes.
                left_ear = calculate_ear(landmarks, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)
                right_ear = calculate_ear(landmarks, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)
                avg_ear = (left_ear + right_ear) / 2.0

                # Draw visual markers on the eyelids for debugging.
                for idx in LEFT_EYE_TOP_BOTTOM + LEFT_EYE_LEFT_RIGHT + RIGHT_EYE_TOP_BOTTOM + RIGHT_EYE_LEFT_RIGHT:
                    pt = landmarks[idx]
                    cx, cy = int(pt.x * frame_width), int(pt.y * frame_height)
                    cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

                # Check if eyes are shut.
                if avg_ear < EAR_THRESHOLD:
                    is_drowsy_this_frame = True

        # Update alert state machine.
        if is_drowsy_this_frame:
            drowsy_counter += 1
        else:
            drowsy_counter = max(0, drowsy_counter - 1)

        # Trigger cloud save routine if thresholds are breached.
        if drowsy_counter > DROWSY_FRAMES_LIMIT and not alert_active:
            alert_active = True
            timestamp = int(time.time())
            filename = f"drowsy_incident_{timestamp}.mp4"

            # Save the contents of the buffer locally.
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
            while frame_buffer:
                out.write(frame_buffer.popleft())
            out.release()
            print(f"🚨 CLOUD AUTOMATION: Eyes closed for 1.5s! Uploading clip: {filename}")

        if drowsy_counter == 0:
            alert_active = False

        # UI warning text overlay on the live stream.
        if drowsy_counter > 10:
            cv2.putText(frame, "ALERT: WAKE UP!", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

        # Display the real-time EAR telemetry on screen.
        cv2.putText(frame, f"Eye Score: {drowsy_counter}", (50, frame_height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Compress the frame into JPEG format for network streaming.
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        with frame_ready:
            latest_frame_bytes = buffer.tobytes()
            frame_ready.notify_all()

    cleanup_stream_resources()


def generate_frames():
    """Yield the latest published JPEG frame to each connected client."""
    initialize_stream_resources()

    last_frame = None
    while capture_running:
        with frame_ready:
            frame_ready.wait_for(lambda: not capture_running or latest_frame_bytes is not None and latest_frame_bytes != last_frame, timeout=1.0)

            if not capture_running:
                break

            if latest_frame_bytes is None or latest_frame_bytes == last_frame:
                continue

            last_frame = latest_frame_bytes
            frame_bytes = latest_frame_bytes

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Endpoint that delivers the real-time MJPEG video stream."""
    try:
        initialize_stream_resources()
    except RuntimeError as exc:
        return Response(str(exc), status=503, mimetype='text/plain')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """The landing dashboard interface for the fleet manager."""
    return """
    <html>
      <head>
        <title>Fleet Executive Control Panel</title>
        <style>
          body { font-family: sans-serif; background: #111; color: white; text-align: center; padding: 20px; }
          .container { max-width: 800px; margin: auto; background: #222; padding: 20px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
          img { width: 100%; max-width: 640px; border: 3px solid #ff4444; border-radius: 5px; }
          h1 { color: #ff4444; margin-bottom: 5px; }
          p { color: #aaa; font-size: 14px; }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>Fleet AI Remote Telemetry</h1>
          <p>Live Monitoring Feed (Device: Laptop B Webcam)</p>
          <img src="/video_feed" />
        </div>
      </body>
    </html>
    """

if __name__ == '__main__':
    # Launch server on port 5000, accessible to all machines on the local Wi-Fi network
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
