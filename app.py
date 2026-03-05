# Import necessary libraries
import base64
from flask import Flask, Response, render_template, send_file, send_from_directory
from flask_sqlalchemy import SQLAlchemy
import cv2
import datetime
import numpy as np
import mediapipe as mp
import os
import landmarker
from collections import deque
from io import BytesIO
import zipfile
from threading import Lock


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

    def __repr__(self) -> str:
        """
        String representation of the Alert object for debugging and display purposes.
        :return: A string representation of the Alert object, including camera number, timestamp, and alert type.
        :rtype: str
        """

        return f"Alert('{self.cam_no}, {self.timestamp}', '{self.alert_type}')"


# Create the database tables
with app.app_context():
    db.create_all()


def find_cameras() -> int:
    """
    Finds the number of connected cameras.
    It attempts to open video capture for sequential indices until it fails, counting the number of successful opens.
    As soon as it encounters an index that cannot be opened, it assumes there are no more cameras and returns the count.

    :return: The number of connected cameras.
    :rtype: int
    """

    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        ret, _ = cap.read()
        cap.release()
        if not ret:
            break
        index += 1
    return index


# Videowriter setup
forucc = cv2.VideoWriter_fourcc(*"mp4v")


# Create output directory for evidence videos if it doesn't exist.
output_dir = os.path.join(landmarker.BASE_DIR, "output")
os.makedirs(output_dir, exist_ok=True)


# Track no-face timers per camera source (int index or URL/path).
t1_by_cam = {}
t1_hand_by_cam = {}
state_by_cam = {}
recording_by_cam = {}
evidence_queue_by_cam = {}  # Optional: To store recent frames for evidence if needed.
alert_evidence_paths = []
state_lock = Lock()


# Function to generate video frames and process them for face detection and landmarking
def generate_frames(cam_key):
    """
    Generates video frames from the specified camera and processes them for face detection and landmarking.

    :param cam_key: The key (index or URL/path) of the camera to process.
    :type cam_key: int or str
    """

    # Define global variables
    global t1_by_cam, t1_hand_by_cam, state_by_cam, recording_by_cam, evidence_queue_by_cam, forucc, alert_evidence_paths

    # Define the start time for FPS calculation.
    start = datetime.datetime.now()

    # Initialize state and timers for this camera if not already done.
    with state_lock:
        # Reset the camera state to IDLE when the camera feed starts.
        state_by_cam[cam_key] = "IDLE"

        # Initialize recording state and evidence queue for this camera.
        recording_by_cam[cam_key] = False
        evidence_queue_by_cam[cam_key] = deque()

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

    def stop_recording():
        """
        Stops recording and clears and saves the evidence queue for the camera.
        """

        with state_lock:
            # Get the queued frames for this camera and clear the queue and recording state.
            queued_frames = list(evidence_queue_by_cam.get(cam_key, deque()))

            # Ensure that there are queued frames to save
            if not queued_frames:
                return

            # Reset recording state and clear the evidence queue for this camera.
            recording_by_cam[cam_key] = False
            evidence_queue_by_cam[cam_key] = deque()

        # Create a video file from the queued frames and save it to the output directory with a timestamped filename.
        release_path = os.path.join(
            output_dir,
            f"evidence_cam{cam_key}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
        )

        height, width = queued_frames[0].shape[:2]
        out = cv2.VideoWriter(
            release_path,
            forucc,
            15.0,
            (width, height),
        )

        # Check if the VideoWriter was successfully opened before attempting to write frames.
        if not out.isOpened():
            print(f"Failed to open VideoWriter for {release_path}")
            return

        # Append each frame in the queued frames to the video file.
        for frame in queued_frames:
            out.write(frame)

        # Release the VideoWriter to ensure the video file is properly saved and closed.
        out.release()

        # Only expose evidence file if video writing succeeded.
        if (
            os.path.exists(release_path)
            and os.path.getsize(release_path) > 0
            and os.path.isfile(release_path)
        ):
            with state_lock:
                base = os.path.basename(release_path)
                if base not in alert_evidence_paths:
                    alert_evidence_paths.append(base)
        else:
            print(f"Failed to save evidence video for camera {cam_key}")

    # Initialize video capture
    cam = cv2.VideoCapture(cam_key)
    attempts = 0

    # Wait until the camera is opened, with a maximum of 5 attempts.
    while not cam.isOpened():
        attempts += 1
        if attempts > 5:
            print(f"Failed to open camera {cam_key} after 5 attempts. Exiting.")
            return "Failed to open camera."
        cam = cv2.VideoCapture(cam_key)
        cv2.waitKey(1000)
    print("Camera is ready")

    # Use Face Detector and Face Landmarker
    try:
        # Use context managers to ensure proper resource management of the detectors and landmarkers.
        with (
            # Context manager for face detector
            landmarker.FaceDetector.create_from_options(
                landmarker.face_detector_options
            ) as face_detector,
            # Context manager for face landmarker
            landmarker.FaceLandmarker.create_from_options(
                landmarker.face_landmark_options
            ) as face_landmarker,
            # Context manager for hand landmarker
            landmarker.HandLandmarker.create_from_options(
                landmarker.hand_landmark_options
            ) as hand_landmarker,
        ):
            # Main loop to process video frames
            while True:
                # Read frame from camera
                ret, frame = cam.read()

                # If frame not grabbed, break the loop
                if not ret:
                    print("Failed to grab frame")
                    break

                # If currently recording for this camera, append the current frame to the evidence queue for this camera.
                with state_lock:
                    if recording_by_cam.get(cam_key) is True:
                        evidence_queue_by_cam[cam_key].append(frame)

                # Edit the frame to make it suitable for processing
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Create a MediaPipe Image object from the frame
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Perform face detection (synchronous)
                face_detection_result = face_detector.detect(mp_image)

                # Ensure that the detection result detects a face
                if not face_detection_result.detections:
                    face_detected = False

                # Ensure that the detection result has a valid score for the detected face before proceeding with landmarking and state updates.
                elif face_detection_result.detections[0].categories[0].score > 0.5:
                    face_detected = True

                # If the detection result does not have a valid score for the detected face, treat it as no face detected.
                else:
                    face_detected = False

                # Perform face landmarking if a face is detected
                if face_detected:
                    # Convert the frame to RGB format for processing with MediaPipe.
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Detect face landmarks and draw them on the current frame.
                    landmark_result = face_landmarker.detect(mp_image)
                    annotated_image = landmarker.draw_face_landmarks_on_image(
                        rgb_frame, landmark_result
                    )

                    # Save the annotated frame and convert it back to BGR format for display and saving.
                    frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

                    # Face is back, so reset the no-face timer/state for this camera.
                    with state_lock:
                        t1_by_cam.pop(cam_key, None)
                        state_by_cam[cam_key] = "IDLE"
                    stop_recording()

                # Handle no face detected scenario
                else:
                    # Implement a timer to check if no face is detected for 3 seconds
                    with state_lock:
                        # Ensure that the timer for exists in the global dictionary for this camera, and if not, initialize it to the current time.
                        if cam_key not in t1_by_cam:
                            t1_by_cam[cam_key] = datetime.datetime.now()
                        no_face_start = t1_by_cam[cam_key]
                        recording_by_cam[cam_key] = True
                        current_state = state_by_cam.get(cam_key)
                    t2 = datetime.datetime.now()

                    # Check if 3 seconds have passed
                    if (
                        t2 - no_face_start >= datetime.timedelta(seconds=3)
                        and current_state != "No Face Detected"
                    ):
                        # Alert the user and save the frame
                        alert_type = "No Face Detected"
                        alert(alert_type, frame)

                        # Update the state to "No Face Detected" and reset the timer for this camera.
                        with state_lock:
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

                    # Save the annotated frame and convert it back to BGR format for display and saving.
                    frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

                    # Implement a timer to check if no face is detected for 3 seconds
                    with state_lock:
                        if cam_key not in t1_hand_by_cam:
                            t1_hand_by_cam[cam_key] = datetime.datetime.now()
                        hand_start = t1_hand_by_cam[cam_key]
                        current_state = state_by_cam.get(cam_key, "")
                    t2 = datetime.datetime.now()

                    # Check if 3 seconds have passed
                    if (
                        t2 - hand_start >= datetime.timedelta(seconds=3)
                        and current_state == "IDLE"
                        and face_detected
                    ):
                        # Alert the user and save the frame
                        alert_type = "Hand Raised"
                        alert(alert_type, frame)

                        # Update the state to "Hand Raised" and reset the timer for this camera.
                        with state_lock:
                            state_by_cam[cam_key] = "Hand Raised"
                            t1_hand_by_cam[cam_key] = t2

                # No hand currently detected; reset timer and recover state.
                else:
                    with state_lock:
                        t1_hand_by_cam.pop(cam_key, None)
                        if state_by_cam.get(cam_key) == "Hand Raised" and face_detected:
                            state_by_cam[cam_key] = "IDLE"

                # Calculate FPS
                time_diff = (datetime.datetime.now() - start).total_seconds()

                # Display the FPS on the frame
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

                # Display the current state on the frame
                with state_lock:
                    state_label = state_by_cam.get(cam_key, "IDLE")

                cv2.putText(
                    frame,
                    f"State: {state_label}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                # Reset FPS calculation time
                start = datetime.datetime.now()

                # Show the frame
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + cv2.imencode(".jpg", frame)[1].tobytes()
                    + b"\r\n"
                )
    finally:
        cam.release()


# Flask routes
@app.route("/")
def index() -> str:
    """
    Renders the main page with video feeds and alert counts for each camera.

    :return: The rendered HTML for the main page.
    :rtype: str
    """

    output = ""
    for i in range(find_cameras()):
        output += f"""
            <h2>Camera {i+1}</h2>
            <img src='/video_feed/{i}' width='100%'>
            <p>{Alert.query.filter_by(cam_no=str(i)).count()} alerts</p>
            <a href="/alerts/{i}" class="btn btn-primary">View Alerts</a><hr>
            <form method="post" action="/clear_alerts/{i}" style="display:inline;">
                <button type="submit" class="btn btn-danger">Clear Alerts</button>
            </form><hr>
        """

    return render_template("index.html", content=output, alerts=Alert.query.all())


@app.route("/video_feed/<path:cam_no>")
def video_feed(cam_no: str) -> Response:
    """
    Route to serve the video feed for a specific camera.

    :param cam_no: The camera number or path to the video feed.

    :return: A streaming response containing the video feed.
    :rtype: Response
    """

    # Allow numeric device indices or full URL/device paths.
    if cam_no.isdigit():
        cam_no = int(cam_no)

    # Return the video feed as a multipart response.
    return app.response_class(
        generate_frames(cam_no), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/clear_alerts", methods=["POST"])
def clear_alerts() -> str:
    """
    Clears all alerts from the database.

    :return: A message confirming that all alerts have been cleared, with a link to go back to the main page.
    :rtype: str
    """

    # Use app context to ensure the database session is available when deleting alerts.
    with app.app_context():
        num_rows_deleted = db.session.query(Alert).delete()
        db.session.commit()
    return f"Cleared {num_rows_deleted} alerts! <a href='/'>Go Back</a>"


@app.route("/clear_alerts/<cam_no>", methods=["POST"])
def clear_alerts_by_cam(cam_no) -> str:
    """
    Clears all alerts for a specific camera from the database.

    :param cam_no: The camera number for which to clear alerts.

    :return: A message confirming that alerts for the specified camera have been cleared, with a link to go back to the main page.
    If the camera number is invalid or an error occurs during deletion, an error message is returned instead of crashing the application.
    :rtype: str
    """

    try:
        # Delete alerts for the specified camera
        with app.app_context():
            num_rows_deleted = Alert.query.filter_by(cam_no=str(cam_no)).delete()
            db.session.commit()

    # Handle any exceptions that may occur during the deletion process and return an error message instead of crashing the application.
    except Exception as e:
        return f"Error clearing alerts for camera {cam_no}: {e}"

    # Send the confirmation message to the user
    return f"Cleared {num_rows_deleted} alerts for camera {cam_no}! <a href='/'>Go Back</a>"


@app.route("/alerts")
def alerts() -> str:
    """
    Renders the alerts page showing all alerts in descending order of timestamp.

    :return: The rendered HTML for the alerts page, containing all alerts and available evidence paths for download.
    :rtype: str
    """

    # Query all alerts from the database in descending order of timestamp and prepare them for display on the alerts page.
    alerts = Alert.query.order_by(Alert.timestamp.desc()).all()
    view_alerts = []

    # Loop through the alerts to prepare them for display, encoding the alert images to base64 for rendering in HTML.
    for alert in alerts:
        view_alerts.append(
            {
                "id": alert.id,
                "timestamp": alert.timestamp,
                "cam_no": alert.cam_no,
                "alert_type": alert.alert_type,
                "alert_image": base64.b64encode(alert.alert_image).decode("utf-8"),
            }
        )

    # Update the evidence paths
    with state_lock:
        evidence_paths = list(alert_evidence_paths)

    # Send the page to the user with all alerts and the available evidence paths for download.
    return render_template(
        "alerts.html", alerts=view_alerts, alert_evidence_paths=evidence_paths
    )


@app.route("/alerts/<cam_no>")
def alerts_by_cam(cam_no: str) -> str:
    """
    Renders the alerts page showing alerts for a specific camera in descending order of timestamp.

    :param cam_no: The camera number for which to display alerts.

    :return: The rendered HTML for the alerts page, containing alerts for the specified camera and available evidence paths for download.
    :rtype: str
    """

    # Query alerts for the specified camera from the database in descending order of timestamp and prepare them for display on the alerts page.
    alerts = (
        Alert.query.filter_by(cam_no=str(cam_no)).order_by(Alert.timestamp.desc()).all()
    )

    view_alerts = []

    # Loop through the alerts to prepare them for display, encoding the alert images to base64 for rendering in HTML.
    for alert in alerts:
        view_alerts.append(
            {
                "id": alert.id,
                "timestamp": alert.timestamp,
                "cam_no": alert.cam_no,
                "alert_type": alert.alert_type,
                "alert_image": base64.b64encode(alert.alert_image).decode("utf-8"),
            }
        )

    # Update the evidence paths
    with state_lock:
        evidence_paths = list(alert_evidence_paths)

    # Send the page to the user with the alerts for the specified camera and the available evidence paths for download.
    return render_template(
        "alerts.html",
        alerts=view_alerts,
        cam_no=cam_no,
        alert_evidence_paths=evidence_paths,
    )


@app.route("/delete_alert/<int:alert_id>", methods=["POST"])
def delete_alert(alert_id: int) -> str:
    """
    Deletes a specific alert from the database.

    :param alert_id: The ID of the alert to be deleted.

    :return: A message confirming that the specified alert has been deleted, with a link to view the remaining alerts.
    If the alert ID is not found or an error occurs during deletion, an error message is returned instead of crashing the application.
    :rtype: str
    """

    with app.app_context():
        # Query the alert by ID
        alert = db.session.get(Alert, alert_id)

        # Ensure the alert exists before attempting to delete it to avoid errors.
        if alert:
            db.session.delete(alert)
            db.session.commit()

        # If the alert was not found, return a message indicating so, instead of attempting to delete and causing an error.
        else:
            return (
                f"Alert with id {alert_id} not found! <a href='/alerts'>View Alerts</a>"
            )

    # Send the user back to the alerts page with a message confirming deletion of the alert.
    return f"Deleted alert with id {alert_id}! <a href='/alerts'>View Alerts</a>"


@app.route("/download/<path:filepath>")
def download_file(filepath: str) -> Response:
    """
    Route to download an alert evidence file.

    :param filepath: The path to the file to be downloaded.

    :return: A response that sends the specified file for download.
    :rtype: Response
    """

    return send_from_directory(output_dir, filepath, as_attachment=True)


@app.route("/download_all_alerts")
def download_all_alerts() -> Response:
    """
    Route to download all alert evidence files as a zip archive.

    :return: A response that sends an in-memory zip file containing all alert evidence files for download.
    :rtype: Response
    """

    # Create an in-memory zip file containing all evidence files in the output directory.
    memory_file = BytesIO()

    # Walk through the output directory and add all files to the zip archive, keeping the folder structure intact.
    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)

                # keeps folder structure inside zip
                arcname = os.path.relpath(file_path, output_dir)
                zf.write(file_path, arcname=arcname)

    # Reset the pointer of the in-memory file to the beginning before sending it for download.
    memory_file.seek(0)

    # Send the in-memory zip file for download with an appropriate filename.
    return send_file(
        memory_file,
        as_attachment=True,
        download_name="all_alert_evidence.zip",
    )


# Run the Flask app
if __name__ == "__main__":
    app.run()
