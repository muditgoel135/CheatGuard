import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult


def draw_hand_landmarks_on_image(
    rgb_image: np.ndarray, detection_result: HandLandmarkerResult
):
    """
    Draws the hand landmarks on the image.

    :param rgb_image: The image on which to draw the hand landmarks.
    :param detection_result: The result of hand landmark detection, to tell the locations of the hand landmarks.
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

    return annotated_image


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
