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


# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = (
    sys.argv[1]
    if len(sys.argv) > 1
    else exit("Please provide a secret key as a command line argument.")
)

# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"
db = SQLAlchemy(app)


# Database model for alerts
class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.datetime.now())
    cam_no = db.Column(db.String(100), nullable=False, default=0)
    alert_type = db.Column(db.String(100), nullable=False)
    alert_image = db.Column(db.LargeBinary, nullable=False)

    def __repr__(self):
        return f"Alert('{self.cam_no}, {self.timestamp}', '{self.alert_type}')"


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
mp.tasks = mp.tasks
mp.tasks.vision = mp.tasks.vision
BaseOptions = mp.tasks.BaseOptions

# Define Face Detector and Face Landmarker
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult

# Paths to the models
path_to_detection_model = "detection_models/face_detection_short_range.tflite"
path_to_landmark_model = "detection_models/face_landmarker.task"
# Track no-face timers per camera source (int index or URL/path).
t1_by_cam = {}


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
        with app.app_context():
            db.session.add(new_alert)
            db.session.commit()

    def draw_landmarks_on_image(
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

    # Set up Face Detector and Landmark options
    detector_options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=path_to_detection_model),
        running_mode=VisionRunningMode.IMAGE,
    )
    landmark_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=path_to_landmark_model),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
    )

    cam = cv2.VideoCapture(cam_no)
    while not cam.isOpened():
        cam = cv2.VideoCapture(0)
        cv2.waitKey(1000)
    print("Camera is ready")

    with FaceDetector.create_from_options(
        detector_options
    ) as detector, FaceLandmarker.create_from_options(landmark_options) as landmarker:
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
            detection_result = detector.detect(mp_image)
            if not detection_result.detections:
                face_detected = False
            elif detection_result.detections[0].categories[0].score > 0.5:
                face_detected = True
            else:
                face_detected = False

            if face_detected:
                # Detect face landmarks and draw them on the current frame.
                landmark_result = landmarker.detect(mp_image)
                annotated_image = draw_landmarks_on_image(rgb_frame, landmark_result)
                frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

            else:
                # Handle no face detected scenario
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
    output = ""
    for i in range(find_cameras()):
        output += f"<h2>Camera {i+1}</h2><img src='/video_feed/{i}' width='100%'><p>{Alert.query.filter_by(cam_no=i).count()} alerts</p><hr>"
    return render_template("index.html", content=output, alerts=Alert.query.all())


@app.route("/video_feed/<path:cam_no>")
def video_feed(cam_no):
    # Allow numeric device indices or full URL/device paths.
    if cam_no.isdigit():
        cam_no = int(cam_no)
    return app.response_class(
        generate_frames(cam_no), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/clear_alerts")
def clear_alerts():
    with app.app_context():
        num_rows_deleted = db.session.query(Alert).delete()
        db.session.commit()
    return f"Cleared {num_rows_deleted} alerts! <a href='/'>Go Back</a>"


@app.route("/alerts")
def alerts():
    alerts = Alert.query.order_by(Alert.timestamp.desc()).all()
    for alert in alerts:
        alert.alert_image = base64.b64encode(alert.alert_image).decode("utf-8")
    return render_template("alerts.html", alerts=alerts)


@app.route("/delete_alert/<int:alert_id>")
def delete_alert(alert_id):
    with app.app_context():
        alert = Alert.query.get(alert_id)
        if alert:
            db.session.delete(alert)
            db.session.commit()
    return f"Deleted alert with id {alert_id}! <a href='/alerts'>View Alerts</a>"


if __name__ == "__main__":
    app.run(debug=True, port=8080)
