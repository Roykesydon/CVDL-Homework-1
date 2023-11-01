import cv2
import matplotlib.pyplot as plt
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from question_4.Sift import Sift

from .MessageBoxController import MessageBoxController


class SiftController:
    def __init__(self, main_window, image_loader):
        self._main_window = main_window

        self._image_loader = image_loader
        self._sift = Sift()

        self._scale = 1
        self._disparity = None
        self._image_right = None

        self._image_1_path = None
        self._image_2_path = None

        self._main_window.loadImage1.clicked.connect(self.on_loadImage1_clicked)
        self._main_window.loadImage2.clicked.connect(self.on_loadImage2_clicked)
        self._main_window.keypoints.clicked.connect(self.on_keypoints_clicked)
        self._main_window.matchedKeypoints.clicked.connect(
            self.on_matchedKeypoints_clicked
        )

    def on_loadImage1_clicked(self):
        self._image_1_path = str(QFileDialog.getOpenFileName(None, "Select Image")[0])

    def on_loadImage2_clicked(self):
        self._image_2_path = str(QFileDialog.getOpenFileName(None, "Select Image")[0])

    def on_keypoints_clicked(self):
        if self._image_1_path is None:
            MessageBoxController.error_message_box("Please load image 1 first")
            return

        img1_gray = self._image_loader.load_from_image_path(
            self._image_1_path, gray=True
        )

        # Get keypoints
        kp1, _ = self._sift.get_keypoints_and_descriptors(img1_gray)

        # Draw keypoints
        img1_gray = self._sift.draw_keypoints(img1_gray, kp1, color=(0, 255, 0))

        # Display images
        img1_gray = cv2.resize(img1_gray, (480, 480))
        cv2.imshow("keypoints", img1_gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_matchedKeypoints_clicked(self):
        if self._image_1_path is None:
            MessageBoxController.error_message_box("Please load image 1 first")
            return

        if self._image_2_path is None:
            MessageBoxController.error_message_box("Please load image 2 first")
            return

        img1_gray = self._image_loader.load_from_image_path(
            self._image_1_path, gray=True
        )
        img2_gray = self._image_loader.load_from_image_path(
            self._image_2_path, gray=True
        )

        img3 = self._sift.get_matched_image(img1_gray, img2_gray)

        # resize img3
        # make the longer side to 640
        if img3.shape[0] > img3.shape[1]:
            img3 = cv2.resize(img3, (int(640 * img3.shape[1] / img3.shape[0]), 640))
        else:
            img3 = cv2.resize(img3, (640, int(640 * img3.shape[0] / img3.shape[1])))

        # Display images
        cv2.imshow("matched keypoints", img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
