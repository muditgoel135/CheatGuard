# Import necessary libraries
import base64
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
import cv2
import datetime
import numpy as np
import mediapipe as mp
import os
import landmarker


# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")

# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"
db = SQLAlchemy(app)


# Database model for alerts
class Alert(db.Model):
    """
    Database model for storing alerts generated.
    """

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False)
    cam_no = db.Column(db.String(100), nullable=False, default=0)
    alert_type = db.Column(db.String(100), nullable=False)
    alert_image = db.Column(db.LargeBinary, nullable=False)

    def __repr__(self):
        return f"Alert('{self.cam_no}, {self.timestamp}', '{self.alert_type}')"


# Create the database tables
with app.app_context():
    db.create_all()


# Find the no. of cameras connected
def find_cameras():
    """
    Finds the number of connected cameras.
    """

    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            cap.release()
            break
        index += 1
    return index


# Mediapipe setup
BaseOptions = mp.tasks.BaseOptions

# Define Face Detector and Face Landmarker
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult

# Define Hand detector
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult

# Paths to the models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path_to_face_detection_model = os.path.join(
    BASE_DIR, "detection_models", "face_detection_short_range.tflite"
)
path_to_landmark_model = os.path.join(
    BASE_DIR, "detection_models", "face_landmarker.task"
)
path_to_hand_landmark_model = os.path.join(
    BASE_DIR, "detection_models", "hand_landmarker.task"
)


# Detection and landmarking options with model paths
face_detector_options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=path_to_face_detection_model),
    running_mode=VisionRunningMode.IMAGE,
)

face_landmark_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=path_to_landmark_model),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
)

hand_landmark_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=path_to_hand_landmark_model),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
)

# Track no-face timers per camera source (int index or URL/path).
t1_by_cam = {}
t1_hand_by_cam = {}
state_by_cam = {}
# evidence_queue_by_cam = {}  # Optional: To store recent frames for evidence if needed.


# Function to generate video frames and process them for face detection and landmarking
def generate_frames(cam_key):
    """
    Generates video frames from the specified camera and processes them for face detection and landmarking.

    :param cam_no: Description
    :type cam_no: int
    """

    global t1_by_cam, t1_hand_by_cam, state_by_cam
    start = datetime.datetime.now()
    state_by_cam[cam_key] = "IDLE"

    def alert(alert_type: str, frame: np.ndarray):
        """
        Saves an alert to the database.

        :param alert_type: Description
        :param frame: Description
        """

        print(f"ALERT: {alert_type} at {datetime.datetime.now()}")

        # Convert frame to bytes
        _, buffer = cv2.imencode(".png", frame)
        alert_image = buffer.tobytes()

        # Create new alert and save to database
        new_alert = Alert(
            alert_type=alert_type,
            alert_image=alert_image,
            cam_no=str(cam_key),
            timestamp=datetime.datetime.now(),
        )

        # Use app context to ensure the database session is available
        with app.app_context():
            db.session.add(new_alert)
            db.session.commit()

    # Initialize video capture
    cam = cv2.VideoCapture(cam_key)
    attempts = 0

    # Wait until the camera is opened, with a maximum of 5 attempts (5 seconds).
    while not cam.isOpened():
        attempts += 1
        if attempts > 5:
            print(f"Failed to open camera {cam_key} after 5 attempts. Exiting.")
            return "Failed to open camera."
        cam = cv2.VideoCapture(0)
        cv2.waitKey(1000)
    print("Camera is ready")

    # Use Face Detector and Face Landmarker
    with FaceDetector.create_from_options(
        face_detector_options
    ) as face_detector, FaceLandmarker.create_from_options(
        face_landmark_options
    ) as face_landmarker, HandLandmarker.create_from_options(
        hand_landmark_options
    ) as hand_landmarker:
        # Main loop to process video frames
        while True:
            # Read frame from camera
            ret, frame = cam.read()

            # If frame not grabbed, break the loop
            if not ret:
                print("Failed to grab frame")
                break

            # Edit the frame to make it suitable for processing
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create a MediaPipe Image object from the frame
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Perform face detection (synchronous)
            face_detection_result = face_detector.detect(mp_image)
            if not face_detection_result.detections:
                face_detected = False
            elif face_detection_result.detections[0].categories[0].score > 0.5:
                face_detected = True
            else:
                face_detected = False

            # Perform face landmarking if a face is detected
            if face_detected:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect face landmarks and draw them on the current frame.
                landmark_result = face_landmarker.detect(mp_image)
                annotated_image = landmarker.draw_face_landmarks_on_image(
                    rgb_frame, landmark_result
                )
                frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

                # Face is back, so reset the no-face timer/state for this camera.
                t1_by_cam.pop(cam_key, None)
                if state_by_cam.get(cam_key) == "No Face Detected":
                    state_by_cam[cam_key] = "IDLE"

            # Handle no face detected scenario
            else:
                # Implement a timer to check if no face is detected for 3 seconds
                if cam_key not in t1_by_cam:
                    t1_by_cam[cam_key] = datetime.datetime.now()
                t2 = datetime.datetime.now()

                # Check if 3 seconds have passed
                if (
                    t2 - t1_by_cam[cam_key] >= datetime.timedelta(seconds=3)
                    and state_by_cam.get(cam_key) != "No Face Detected"
                ):
                    # Alert the user and save the frame
                    alert_type = "No Face Detected"
                    alert(alert_type, frame)
                    state_by_cam[cam_key] = "No Face Detected"
                    t1_by_cam[cam_key] = t2

            # Perform hand landmarking to check for raised hand
            hand_landmark_result = hand_landmarker.detect(mp_image)
            if hand_landmark_result.hand_landmarks and (
                hand_landmark_result.handedness[0][0].score > 0.5
            ):
                # If hand landmarks are detected, draw them on the frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated_image = landmarker.draw_hand_landmarks_on_image(
                    rgb_frame, hand_landmark_result
                )
                frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

                # Implement a timer to check if no face is detected for 3 seconds
                if cam_key not in t1_hand_by_cam:
                    t1_hand_by_cam[cam_key] = datetime.datetime.now()
                t2 = datetime.datetime.now()

                # Check if 3 seconds have passed
                if (
                    t2 - t1_hand_by_cam[cam_key] >= datetime.timedelta(seconds=3)
                    and state_by_cam.get(cam_key, "") == "IDLE"
                    and face_detected
                ):
                    # Alert the user and save the frame
                    alert_type = "Hand Raised"
                    alert(alert_type, frame)
                    state_by_cam[cam_key] = "Hand Raised"
                    t1_hand_by_cam[cam_key] = t2
            else:
                # No hand currently detected; reset timer and recover state.
                t1_hand_by_cam.pop(cam_key, None)
                if state_by_cam.get(cam_key) == "Hand Raised" and face_detected:
                    state_by_cam[cam_key] = "IDLE"

            # Calculate and display FPS on the frame
            time_diff = (datetime.datetime.now() - start).total_seconds()
            cv2.putText(
                frame,
                f"FPS: {1/time_diff:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            start = datetime.datetime.now()

            # Display the current state on the frame
            cv2.putText(
                frame,
                f"State: {state_by_cam.get(cam_key, 'IDLE')}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Show the frame
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + cv2.imencode(".jpg", frame)[1].tobytes()
                + b"\r\n"
            )


# Flask routes
@app.route("/")
def index():
    """
    Renders the main page with video feeds and alert counts for each camera.
    """

    output = ""
    for i in range(find_cameras()):
        output += f"""
            <h2>Camera {i+1}</h2>
            <img src='/video_feed/{i}' width='100%'>
            <p>{Alert.query.filter_by(cam_no=str(i)).count()} alerts</p>
            <a href="/alerts/{i}" class="btn btn-primary">View Alerts</a><hr>
            <a href="/clear_alerts/{i}" class="btn btn-danger">Clear Alerts</a><hr>
        """

    return render_template("index.html", content=output, alerts=Alert.query.all())


@app.route("/video_feed/<path:cam_no>")
def video_feed(cam_no):
    """
    Route to serve the video feed for a specific camera.

    :param cam_no: The camera number or path to the video feed.
    """

    # Allow numeric device indices or full URL/device paths.
    if cam_no.isdigit():
        cam_no = int(cam_no)

    # Return the video feed as a multipart response.
    return app.response_class(
        generate_frames(cam_no), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/clear_alerts")
def clear_alerts():
    """
    Clears all alerts from the database.
    """

    # Use app context to ensure the database session is available when deleting alerts.
    with app.app_context():
        num_rows_deleted = db.session.query(Alert).delete()
        db.session.commit()
    return f"Cleared {num_rows_deleted} alerts! <a href='/'>Go Back</a>"


@app.route("/clear_alerts/<cam_no>")
def clear_alerts_by_cam(cam_no):
    """
    Clears all alerts for a specific camera from the database.

    :param cam_no: The camera number for which to clear alerts.
    """

    with app.app_context():
        num_rows_deleted = Alert.query.filter_by(cam_no=str(cam_no)).delete()
        db.session.commit()
    return f"Cleared {num_rows_deleted} alerts for camera {cam_no}! <a href='/'>Go Back</a>"


@app.route("/alerts")
def alerts():
    """
    Renders the alerts page showing all alerts in descending order of timestamp.
    """

    alerts = Alert.query.order_by(Alert.timestamp.desc()).all()
    for alert in alerts:
        alert.alert_image = base64.b64encode(alert.alert_image).decode("utf-8")
    return render_template("alerts.html", alerts=alerts)


@app.route("/alerts/<cam_no>")
def alerts_by_cam(cam_no):
    """
    Renders the alerts page showing alerts for a specific camera in descending order of timestamp.

    :param cam_no: The camera number for which to display alerts.
    """

    alerts = (
        Alert.query.filter_by(cam_no=str(cam_no)).order_by(Alert.timestamp.desc()).all()
    )
    for alert in alerts:
        alert.alert_image = base64.b64encode(alert.alert_image).decode("utf-8")
    return render_template("alerts.html", alerts=alerts, cam_no=cam_no)


@app.route("/delete_alert/<int:alert_id>")
def delete_alert(alert_id):
    """
    Deletes a specific alert from the database.

    :param alert_id: The ID of the alert to be deleted.
    """

    with app.app_context():
        alert = Alert.query.get(alert_id)
        if alert:
            db.session.delete(alert)
            db.session.commit()
    return f"Deleted alert with id {alert_id}! <a href='/alerts'>View Alerts</a>"


# Run the Flask app
if __name__ == "__main__":
    app.run(port=8080)
