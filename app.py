# Import necessary libraries
import base64
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
import sys
import cv2
import datetime
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils
import os


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
path_to_face_detection_model = "detection_models/face_detection_short_range.tflite"
path_to_landmark_model = "detection_models/face_landmarker.task"
path_to_hand_landmark_model = "detection_models/hand_landmarker.task"


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


# Function to generate video frames and process them for face detection and landmarking
def generate_frames(cam_no):
    """
    Generates video frames from the specified camera and processes them for face detection and landmarking.

    :param cam_no: Description
    :type cam_no: int
    """

    global t1_by_cam
    cam_key = cam_no

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
            cam_no=cam_no,
            timestamp=datetime.datetime.now(),
        )

        # Use app context to ensure the database session is available
        with app.app_context():
            db.session.add(new_alert)
            db.session.commit()

    def draw_face_landmarks_on_image(
        rgb_image: np.ndarray, detection_result: FaceLandmarkerResult
    ):
        """
        Draws the face landmarks on the image.

        :param rgb_image: Description
        :param detection_result: Description
        """

        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks on the image.
            # Tesselation
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
            )

            # Contours
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
            )

            # Irises
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
            )

            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
            )

        return annotated_image

    def draw_hand_landmarks_on_image(
        rgb_image: np.ndarray, detection_result: HandLandmarkerResult
    ):
        """
        Draws the hand landmarks on the image.

        :param rgb_image: Description
        :param detection_result: Description
        """

        hand_landmarks_list = detection_result.hand_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # Draw the hand landmarks on the image.
            drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks,
                # connections=vision.HandLandmarksConnections.HAND_LANDMARKS,
                landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=drawing_styles.get_default_hand_connections_style(),
            )

        return annotated_image

    # Initialize video capture
    cam = cv2.VideoCapture(cam_no)
    attempts = 0
    while not cam.isOpened():
        attempts += 1
        if attempts > 5:
            print(f"Failed to open camera {cam_no} after 5 attempts. Exiting.")
            sys.exit(1)
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
                annotated_image = draw_face_landmarks_on_image(
                    rgb_frame, landmark_result
                )
                frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

            # Handle no face detected scenario
            else:
                # Implement a timer to check if no face is detected for 3 seconds
                if cam_key not in t1_by_cam:
                    t1_by_cam[cam_key] = datetime.datetime.now()
                t2 = datetime.datetime.now()

                # Check if 3 seconds have passed
                if t2 - t1_by_cam[cam_key] >= datetime.timedelta(seconds=3):
                    # Alert the user and save the frame
                    alert_type = "No Face Detected"
                    alert(alert_type, frame)
                    t1_by_cam[cam_key] = t2

            # Perform hand landmarking to check for raised hand
            hand_landmark_result = hand_landmarker.detect(mp_image)
            if hand_landmark_result.hand_landmarks:
                # If hand landmarks are detected, draw them on the frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                annotated_image = draw_hand_landmarks_on_image(
                    rgb_frame, hand_landmark_result
                )
                frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

                # Implement a timer to check if no face is detected for 3 seconds
                if cam_key not in t1_hand_by_cam:
                    t1_hand_by_cam[cam_key] = datetime.datetime.now()
                t2 = datetime.datetime.now()

                # Check if 3 seconds have passed
                if t2 - t1_hand_by_cam[cam_key] >= datetime.timedelta(seconds=3):
                    # Alert the user and save the frame
                    alert_type = "Hand Raised"
                    alert(alert_type, frame)
                    t1_hand_by_cam[cam_key] = t2

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
        output += f"<h2>Camera {i+1}</h2><img src='/video_feed/{i}' width='100%'><p>{Alert.query.filter_by(cam_no=i).count()} alerts</p><hr>"
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
    return app.response_class(
        generate_frames(cam_no), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/clear_alerts")
def clear_alerts():
    """
    Clears all alerts from the database.
    """

    with app.app_context():
        num_rows_deleted = db.session.query(Alert).delete()
        db.session.commit()
    return f"Cleared {num_rows_deleted} alerts! <a href='/'>Go Back</a>"


@app.route("/alerts")
def alerts():
    """
    Renders the alerts page showing all alerts in descending order of timestamp.
    """

    alerts = Alert.query.order_by(Alert.timestamp.desc()).all()
    for alert in alerts:
        alert.alert_image = base64.b64encode(alert.alert_image).decode("utf-8")
    return render_template("alerts.html", alerts=alerts)


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
