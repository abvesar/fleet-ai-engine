import cv2
import time
import math
import atexit
import threading
import socket
from collections import deque
from flask import Flask, Response, jsonify, request
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
STREAM_WIDTH = 640
STREAM_HEIGHT = 360
PROCESSING_SCALE = 0.5
JPEG_QUALITY = 70
ANALYSIS_INTERVAL = 3
frame_buffer = deque(maxlen=fps * 5)
latest_frame_bytes = None
capture_thread = None
capture_running = False
stream_lock = threading.Lock()
frame_ready = threading.Condition(stream_lock)
analysis_frame_count = 0


def build_placeholder_frame():
    """Create a lightweight frame so streaming clients receive bytes immediately."""
    frame = cv2.cvtColor(
        cv2.resize(
            cv2.UMat(90, 160, cv2.CV_8UC3, (20, 20, 20)).get(),
            (STREAM_WIDTH, STREAM_HEIGHT)
        ),
        cv2.COLOR_BGR2RGB
    )
    cv2.putText(frame, "Starting camera feed...", (20, STREAM_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buffer.tobytes() if ret else b""


def initialize_stream_resources():
    """Open the camera and create the face mesh detector once."""
    global cap, face_mesh, frame_width, frame_height, fps, frame_buffer, buffer_length, DROWSY_FRAMES_LIMIT, drowsy_counter, alert_active, capture_thread, capture_running

    if cap is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            cap = None
            raise RuntimeError("Unable to open camera 0")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, STREAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STREAM_HEIGHT)

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


def get_lan_ip():
    """Best-effort local IP address for remote-device access."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def capture_loop():
    """Read frames, run detection, and publish the latest encoded JPEG for all clients."""
    global drowsy_counter, alert_active, latest_frame_bytes, analysis_frame_count

    while capture_running and cap is not None and cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Store clean frame in the rolling incident buffer.
        frame_buffer.append(frame.copy())

        is_drowsy_this_frame = False
        analysis_frame_count = (analysis_frame_count + 1) % ANALYSIS_INTERVAL

        if analysis_frame_count == 0:
            processing_frame = cv2.resize(frame, None, fx=PROCESSING_SCALE, fy=PROCESSING_SCALE)

            # Convert BGR image to RGB for MediaPipe processing.
            rgb_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

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

            # Update alert state machine only when analysis runs.
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
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ret:
            continue

        with frame_ready:
            latest_frame_bytes = buffer.tobytes()
            frame_ready.notify_all()

    cleanup_stream_resources()


def generate_frames():
    """Yield the latest published JPEG frame to each connected client."""
    initialize_stream_resources()

    last_frame = build_placeholder_frame()
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + last_frame + b'\r\n')

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

    response = Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


@app.route('/alert_status')
def alert_status():
    """Expose the current distraction alert state for the dashboard."""
    return jsonify({"alert_active": alert_active, "drowsy_counter": drowsy_counter})

@app.route('/')
def index():
    """The landing dashboard interface for the fleet manager."""
    base_url = request.host_url.rstrip("/")
    lan_ip = get_lan_ip()
    html = """
    <html>
      <head>
        <title>Fleet Executive Control Panel</title>
        <style>
          body { font-family: sans-serif; background: #111; color: white; text-align: center; padding: 20px; }
          .container { max-width: 800px; margin: auto; background: #222; padding: 20px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
          img { width: 100%; max-width: 640px; border: 3px solid #ff4444; border-radius: 5px; }
          h1 { color: #ff4444; margin-bottom: 5px; }
          p { color: #aaa; font-size: 14px; }
                    #alert-banner { display: none; margin: 15px auto; max-width: 640px; padding: 12px 16px; border-radius: 8px; background: #8b0000; color: white; font-weight: bold; }
        </style>
      </head>
      <body>
                <div class="container">
                    <h1>Fleet AI Remote Telemetry</h1>
                    <p>Live Monitoring Feed (Device: Laptop B Webcam)</p>
                    <p>Open this on another device: __BASE_URL__</p>
                    <p>LAN IP: __LAN_IP__</p>
                                        <div id="alert-banner">Driver distraction detected.</div>
                    <img src="/video_feed" />
                </div>
                                <script>
                                    let previousAlertState = false;
                                    let notificationPermissionRequested = false;

                                    function playAlertSound() {
                                        try {
                                            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                                            const oscillator = audioContext.createOscillator();
                                            const gainNode = audioContext.createGain();
                                            
                                            oscillator.connect(gainNode);
                                            gainNode.connect(audioContext.destination);
                                            
                                            // Play a high-frequency beep for alert
                                            oscillator.frequency.value = 1000; // Hz
                                            oscillator.type = 'sine';
                                            
                                            gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
                                            gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
                                            
                                            oscillator.start(audioContext.currentTime);
                                            oscillator.stop(audioContext.currentTime + 0.3);
                                            
                                            // Play second beep after a short delay
                                            setTimeout(() => {
                                                const osc2 = audioContext.createOscillator();
                                                const gain2 = audioContext.createGain();
                                                osc2.connect(gain2);
                                                gain2.connect(audioContext.destination);
                                                osc2.frequency.value = 1200;
                                                osc2.type = 'sine';
                                                gain2.gain.setValueAtTime(0.3, audioContext.currentTime);
                                                gain2.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
                                                osc2.start(audioContext.currentTime);
                                                osc2.stop(audioContext.currentTime + 0.3);
                                            }, 350);
                                        } catch (e) {
                                            console.warn('Audio context not available:', e);
                                        }
                                    }

                                    async function pollAlertStatus() {
                                        try {
                                            const response = await fetch('/alert_status', { cache: 'no-store' });
                                            if (!response.ok) {
                                                return;
                                            }

                                            const data = await response.json();
                                            const banner = document.getElementById('alert-banner');
                                            banner.style.display = data.alert_active ? 'block' : 'none';

                                            if (!notificationPermissionRequested && 'Notification' in window && Notification.permission === 'default') {
                                                notificationPermissionRequested = true;
                                                Notification.requestPermission();
                                            }

                                            if (data.alert_active && !previousAlertState) {
                                                playAlertSound();
                                                if ('Notification' in window && Notification.permission === 'granted') {
                                                    new Notification('Fleet AI Alert', { body: 'Driver distraction detected on the live feed.' });
                                                } else {
                                                    window.alert('Fleet AI Alert: Driver distraction detected on the live feed.');
                                                }
                                            }

                                            previousAlertState = data.alert_active;
                                        } catch (error) {
                                            console.error('Alert polling failed', error);
                                        }
                                    }

                                    pollAlertStatus();
                                    setInterval(pollAlertStatus, 1000);
                                </script>
            </body>
        </html>
        """
    return html.replace("__BASE_URL__", base_url).replace("__LAN_IP__", lan_ip)


if __name__ == '__main__':
        # Launch server on port 5000, accessible to all machines on the local Wi-Fi network
        print("Server running on http://0.0.0.0:5000")
        print(f"Try http://{get_lan_ip()}:5000 from another device on the same network")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
