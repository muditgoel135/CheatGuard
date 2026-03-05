# Import necessary libraries
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult


# Mediapipe setup
BaseOptions = mp.tasks.BaseOptions


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


# Define Face Detector and Face Landmarker
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions


# Define Hand Landmarker
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions


# Mediapipe Detection and landmarking options with model paths
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


def draw_hand_landmarks_on_image(
    rgb_image: np.ndarray, detection_result: HandLandmarkerResult
) -> np.ndarray:
    """
    Draws the hand landmarks from the given detection result on the given image.

    :param rgb_image: The image on which to draw the hand landmarks.
    :param detection_result: The result of hand landmark detection, to tell the locations of the hand landmarks.

    :return: The annotated image with hand landmarks drawn on it.
    :rtype: np.ndarray
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
            connections=vision.HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=drawing_styles.get_default_hand_connections_style(),
        )

    # Return the annotated image with hand landmarks drawn on it.
    return annotated_image


def draw_face_landmarks_on_image(
    rgb_image: np.ndarray, detection_result: FaceLandmarkerResult
) -> np.ndarray:
    """
    Draws the face landmarks from the given detection result on the given image.

    :param rgb_image: The image on which to draw the face landmarks.
    :param detection_result: The result of face landmark detection, to tell the locations of the face landmarks.

    :return: The annotated image with face landmarks drawn on it.
    :rtype: np.ndarray
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

    # Return the annotated image with face landmarks drawn on it.
    return annotated_image
