import sys
import time

import cv2 as cv
import keras.models as km
import numpy
from PyQt5 import uic
from PyQt5.QtCore import QSize, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import *

from src.hand_tracker.hand_landmark_tracker import HandLandmarkDetector
from src.hand_tracker.utils.drawing_utils import draw_hand_landmarks

# from src.utils import landmark_data_processors
from src.utils.landmark_data_processors import pre_process_landmark

# import tensorflow as tf


class app_gui(QMainWindow):
    def __init__(self):
        super(app_gui, self).__init__()
        uic.loadUi("./asserts/ui/app.ui", self)
        self.output_string = ""

        self.text_area: QTextEdit = self.findChild(QTextEdit, "text_panel")
        self.clear_btn: QPushButton = self.findChild(
            QPushButton, "clear_button"
        )
        self.video_screen: QLabel = self.findChild(QLabel, "video_panel")
        self.clear_btn.clicked.connect(self.clear_text)
        self.video_feed = Video_feed()
        self.video_feed.start()
        self.video_feed.image_update.connect(self.update_video_screen)
        self.video_feed.char_predicted.connect(self.update_output_string)

    def video_stop(self):
        self.video_feed.stop()

    def clear_text(self):
        self.output_string = ""
        self.text_area.setText("")

    def update_video_screen(self, image):
        self.video_screen.setPixmap(
            QPixmap.fromImage(image).scaled(
                QSize(self.video_screen.width(), self.video_screen.height())
            )
        )
        self.show()

    def update_output_string(self, char):
        if len(self.output_string) != 0 and char == "del":
            self.output_string = self.output_string[:-1]
        elif char == "spc":
            self.output_string += " "
        else:
            self.output_string += char

        self.text_area.setText(self.output_string)


class Video_feed(QThread):
    image_update = pyqtSignal(QImage)
    char_predicted = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.model = km.load_model("./asserts/models/model_0_2.keras")

        self.frame = cv.VideoCapture(0)
        self.detector = HandLandmarkDetector()
        self.thread_active = True
        self.delay = 4
        self.labels = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "spc",
            "del",
        ]

        self.s_time = time.time()
        self.c_time = 0

    def run(
        self,
    ):
        while True:
            ret, img = self.frame.read()
            data = self.detector.detect(img)

            if data:
                # capture_data(data[0]["lm_list"], 5, 0.5, 400, 28, PATH)
                draw_hand_landmarks(img, data[0]["lm_list"])
                if self.c_time - self.s_time >= self.delay:
                    p_data = numpy.array(
                        pre_process_landmark(data[0]["lm_list"])
                    )
                    p_data = numpy.resize(p_data, (1, 42))
                    # print(p_data)
                    t = self.model.predict(p_data)
                    u = numpy.argmax(t[0])
                    self.char_predicted.emit(self.labels[u])
                    self.s_time = self.c_time
                self.c_time = time.time()
            else:
                self.s_time = time.time()
            qformat_image = QImage(
                img.data, img.shape[1], img.shape[0], QImage.Format_BGR888
            )
            self.image_update.emit(qformat_image)

    def stop(self):
        self.thread_active = False
        self.frame.release()
        self.quit()


app = QApplication(sys.argv)
win = app_gui()
app.exec_()
