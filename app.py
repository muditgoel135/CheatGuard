# Import necessary libraries
from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy
import sys
import cv2
import datetime
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles
from mediapipe.tasks.python.vision import drawing_utils


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

# Initialize camera
cam0 = cv2.VideoCapture(0)

# Mediapipe setup
mp.tasks = mp.tasks
mp.tasks.vision = mp.tasks.vision
BaseOptions = mp.tasks.BaseOptions
mp_image = None

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
face_detected = False


# Create a face detector instance with the live stream mode:
def print_result(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
    """
    Callback function to print the detection result.
    Determines whether a face is detected based on the detection score.

    :param result: Description
    :type result: FaceDetectorResult
    :param output_image: Description
    :type output_image: mp.Image
    :param timestamp_ms: Description
    :type timestamp_ms: int
    """

    global face_detected
    # print(
    #     f"Face detection result at {timestamp_ms} ms: {result.detections[0].categories[0].score if result.detections else 'No faces detected.'}"
    # )
    if not result.detections:
        face_detected = False
        return
    
    if result.detections[0].categories[0].score > 0.5:
        face_detected = True
    else:
        face_detected = False


def draw_landmarks_on_image(rgb_image, detection_result):
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


def landmark_print_result(
    result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
    """
    Draws the face landmarks on the image and updates the global frame variable.

    :param result: Description
    :type result: FaceLandmarkerResult
    :param output_image: Description
    :type output_image: mp.Image
    :param timestamp_ms: Description
    :type timestamp_ms: int
    """

    global mp_image, frame
    # print(
    #     f"Face landmark result at {timestamp_ms} ms: {len(result.face_landmarks)} face landmarks detected."
    # )
    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), result)
    bgr_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    frame = bgr_annotated_image


# Set up Face Detector options
detector_options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=path_to_detection_model),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
)

# Use OpenCV’s VideoCapture to start capturing from the webcam.
while not cam0.isOpened():
    cam0 = cv2.VideoCapture(0)
    cv2.waitKey(1000)
print("Camera is ready")


# Main loop to process video frames
while True:
    # Read frame from camera
    ret, frame = cam0.read()
    # If frame not grabbed, break the loop
    if not ret:
        print("Failed to grab frame")
        break

    # Edit the frame to make it suitable for processing
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Save the width and height of the frame
    img_h, img_w = frame.shape[:2]

    # Create a MediaPipe Image object from the frame
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Perform face detection
    with FaceDetector.create_from_options(detector_options) as detector:
        detector.detect_async(
            mp_image, int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        )

    # If a face is detected, perform face landmarking
    if face_detected:
        # Create a face landmarker instance with the live stream mode:
        landmark_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=path_to_landmark_model),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=landmark_print_result,
            num_faces=1,
        )

        # Use the face landmarker to detect face landmarks from the input image.
        with FaceLandmarker.create_from_options(landmark_options) as landmarker:
            landmarker.detect_async(
                mp_image, int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            )
    else:
        # Handle no face detected scenario
        # Implement a timer to check if no face is detected for 3 seconds
        if not "t1" in globals():
            t1 = datetime.datetime.now()
        t2 = datetime.datetime.now()

        # Check if 3 seconds have passed
        if t2 - t1 >= datetime.timedelta(seconds=3):
            # Alert the user and save the frame
            print("No face detected for 3 seconds.")
            cv2.imwrite(f"alerts/no_face_detected/{t2.timestamp()}.png", frame)
            t1 = t2

    # Show the frame
    cv2.imshow("Webcam", frame)

    # Exit on ESC key
    k = cv2.waitKey(1)
    if k % 256 == 27:  # ESC pressed
        print("Escape hit, closing...")
        break


# Release resources
cam0.release()
cv2.destroyAllWindows()
