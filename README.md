# Fleet AI Engine

Fleet AI Engine is a local Flask-based prototype that streams webcam video and runs drowsiness detection with MediaPipe face landmarks.

## What It Does

- Opens the local webcam.
- Detects faces and eye landmarks with MediaPipe.
- Calculates an eye-aspect score to estimate drowsiness.
- Streams the processed camera feed through a Flask MJPEG endpoint.
- Saves a short incident clip when drowsiness persists long enough.

## Files

- `fleet_server.py`: Flask server and live drowsiness-monitoring stream.
- `fleet_prototype.py`: Prototype script for the broader fleet AI workflow.

## Requirements

- Python 3.10+
- `flask`
- `opencv-python`
- `mediapipe`

The project also uses `ultralytics` and related dependencies for model-based features.

## Setup

If you are using the included virtual environment:

```powershell
& .venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you do not have a `requirements.txt` yet, install the needed packages directly:

```powershell
& .venv\Scripts\python.exe -m pip install flask opencv-python mediapipe ultralytics
```

## Run

Start the server with:

```powershell
& .venv\Scripts\python.exe fleet_server.py
```

Open the dashboard in your browser:

```text
http://127.0.0.1:5000
```

## Controls

- Press `q` in the camera window to stop the stream.

## Notes

- The server uses the default camera at index `0`.
- If the camera is already in use, the stream will return an error message instead of crashing.
- The current implementation is optimized for a single local camera stream.
- For remote access, open the LAN IP shown on the dashboard or startup log, for example `http://<your-ip>:5000`.
- If another device still cannot connect, allow Python through Windows Firewall or the local network firewall.