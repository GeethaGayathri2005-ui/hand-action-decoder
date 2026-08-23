import copy
from typing import Optional

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker


class HandLandmarkDetector:
    __BASE_MODEL: str = "./asserts/models/hand_landmarker.task"

    def __init__(
        self,
        number_of_hands: int = 2,
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: Optional[str] = __BASE_MODEL,
    ) -> None:

        self.num_hands: int = number_of_hands
        self.min_hand_detection_confidence: float = (
            min_hand_detection_confidence
        )
        self.min_hand_presence_confidence: float = min_hand_presence_confidence
        self.min_tracking_confidence: float = min_tracking_confidence
        self.model = model_path

        # initializing mediapipe hand detector using CPU delegate for Windows compatibility
        base_options = python.BaseOptions(
            self.model, delegate=python.BaseOptions.Delegate.CPU
        )
        options = vision.HandLandmarkerOptions(
            base_options,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_hand_detection_confidence,
            min_hand_presence_confidence=self.min_hand_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self.hand_detector = HandLandmarker.create_from_options(options)

    @staticmethod
    def __calc_landmarks_and_bounding_rect(landmarks, image):
        image_height, image_width = image.shape[0], image.shape[1]
        lm_list = []
        for landmark in landmarks:
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            lm_list.append([landmark_x, landmark_y])

        lm_list = np.array(lm_list)
        x, y, w, h = cv.boundingRect(lm_list)
        return lm_list, [x, y, x + w, y + h]

    def detect(
        self,
        image: np.ndarray,
        handedness: list = ["Right", "Left"],
    ) -> list[dict]:
        temp_image = copy.deepcopy(image)
        mp_image = mp.Image(
            mp.ImageFormat.SRGB, cv.cvtColor(temp_image, cv.COLOR_BGR2RGB)
        )
        result = self.hand_detector.detect(mp_image)
        hands = result.handedness
        landmarks = result.hand_landmarks
        landmark_data = []
        if not hands:
            return landmark_data

        for idx in range(len(hands)):
            if hands[idx][0].category_name in handedness:
                hand_landmarks_list = landmarks[idx]
                hand = hands[idx][0].category_name
                lm_list, bb_rect = self.__calc_landmarks_and_bounding_rect(
                    hand_landmarks_list, image
                )
                output_data = {
                    "handedness": hand,
                    "lm_list": lm_list,
                    "bb_rect": bb_rect,
                }
                landmark_data.append(output_data)

        return landmark_data
