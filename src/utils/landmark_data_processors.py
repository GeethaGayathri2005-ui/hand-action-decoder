import copy
import csv
import itertools
from typing import Any, Optional

import numpy as np


def pre_process_landmark(landmark_list, tag: Optional[int] = None):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    if tag is not None:
        temp_landmark_list.insert(0, tag)

    return temp_landmark_list


def logging_csv(path, landmarks: list):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(landmarks)
