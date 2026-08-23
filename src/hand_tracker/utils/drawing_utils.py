import cv2 as cv
import numpy as np

_BGR_CHANNELS = 3

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)


#
# WRIST = 0
#
# THUMB_CMC = 1
# THUMB_MCP = 2
# THUMB_IP = 3
# THUMB_TIP = 4
#
# INDEX_FINGER_MCP = 5
# INDEX_FINGER_PIP = 6
# INDEX_FINGER_DIP = 7
# INDEX_FINGER_TIP = 8
#
# MIDDLE_FINGER_MCP = 9
# MIDDLE_FINGER_PIP = 10
# MIDDLE_FINGER_DIP = 11
# MIDDLE_FINGER_TIP = 12
#
# RING_FINGER_MCP = 13
# RING_FINGER_PIP = 14
# RING_FINGER_DIP = 15
# RING_FINGER_TIP = 16
#
# PINKY_MCP = 17
# PINKY_PIP = 18
# PINKY_DIP = 19
# PINKY_TIP = 20


_HAND_PALM_CONNECTIONS = ((0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17))
_HAND_THUMB_CONNECTIONS = ((1, 2), (2, 3), (3, 4))
_HAND_INDEX_FINGER_CONNECTIONS = ((5, 6), (6, 7), (7, 8))
_HAND_MIDDLE_FINGER_CONNECTIONS = ((9, 10), (10, 11), (11, 12))
_HAND_RING_FINGER_CONNECTIONS = ((13, 14), (14, 15), (15, 16))
_HAND_PINKY_FINGER_CONNECTIONS = ((17, 18), (18, 19), (19, 20))
_HAND_CONNECTIONS = frozenset().union(
    *[
        _HAND_PALM_CONNECTIONS,
        _HAND_THUMB_CONNECTIONS,
        _HAND_INDEX_FINGER_CONNECTIONS,
        _HAND_MIDDLE_FINGER_CONNECTIONS,
        _HAND_RING_FINGER_CONNECTIONS,
        _HAND_PINKY_FINGER_CONNECTIONS,
    ]
)

LINE_THICKNESS = 1
CIRCLE_RADIUS = 5
THICKNESS_DOT = -1


def draw_hand_landmarks(
    image: np.ndarray, landmarks: np.ndarray, indexs: list = []
):
    connections = _HAND_CONNECTIONS
    if image.shape[2] != _BGR_CHANNELS:
        raise ValueError("Input image must contain three channel bgr data.")
    idx_to_coordinates = {}
    if not indexs:
        for idx, landmark in enumerate(landmarks):
            idx_to_coordinates[idx] = landmark
    else:
        for idx in indexs:
            idx_to_coordinates[idx] = landmarks[idx]
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]

        if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:

            cv.line(
                image,
                idx_to_coordinates[start_idx],
                idx_to_coordinates[end_idx],
                WHITE_COLOR,
                LINE_THICKNESS,
            )

        for idx, landmark_px in idx_to_coordinates.items():
            cv.circle(
                image, landmark_px, CIRCLE_RADIUS, RED_COLOR, THICKNESS_DOT
            )


def draw_boundingbox(image: np.ndarray, bb_data: np.array):
    pass


class Data_card:
    pass
