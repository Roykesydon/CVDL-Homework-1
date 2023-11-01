import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from application.utils.InputChecker import InputChecker
from question_3.StereoDisparity import StereoDisparity


class StereoDisparityController:
    def __init__(self, main_window, image_loader):
        self._main_window = main_window

        self._image_loader = image_loader
        self._stereo_disparity = StereoDisparity()

        self._scale = 1
        self._disparity = None
        self._image_right = None

        self._main_window.stereoDisparityMap.clicked.connect(
            self.on_stereoDisparityMap_clicked
        )

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and param == "imgL":
            real_y, real_x = x * self._scale, y * self._scale

            # get img_right_x, img_right_y
            img_right_coordinates = (
                self._stereo_disparity.get_img_right_coresponding_coordinates(
                    int(real_x), int(real_y), self._disparity
                )
            )

            if img_right_coordinates is None:
                print("No corresponding coordinates")
                return

            img_right_x, img_right_y = img_right_coordinates

            rescale_img_right_x, rescale_img_right_y = (
                int(img_right_x / self._scale),
                int(img_right_y / self._scale),
            )

            draw_img_right = self._stereo_disparity.draw_circle_on_img(
                self._image_right,
                rescale_img_right_x,
                rescale_img_right_y,
                4,
                (0, 255, 0),
            )

            cv2.imshow("imgR", draw_img_right)

    @InputChecker.check_image_left_and_right_empty
    def on_stereoDisparityMap_clicked(self):
        img_left_path = self._image_loader.get_image_left_path()
        img_right_path = self._image_loader.get_image_right_path()

        img_left_gray = self._image_loader.load_from_image_path(
            img_left_path, gray=True
        )
        img_right_gray = self._image_loader.load_from_image_path(
            img_right_path, gray=True
        )

        img_left = cv2.imread(img_left_path)
        img_right = cv2.imread(img_right_path)

        self._stereo_disparity.set_img_left(img_left_gray)
        self._stereo_disparity.set_img_right(img_right_gray)

        disparity = self._stereo_disparity.compute_disparity()
        self._disparity = disparity.copy()

        longest_shape = max(disparity.shape[:2])

        self._scale = longest_shape / 480

        disparity = cv2.resize(
            disparity,
            (
                int(disparity.shape[1] / self._scale),
                int(disparity.shape[0] / self._scale),
            ),
        )
        img_left = cv2.resize(
            img_left,
            (
                int(img_left.shape[1] / self._scale),
                int(img_left.shape[0] / self._scale),
            ),
        )
        img_right = cv2.resize(
            img_right,
            (
                int(img_right.shape[1] / self._scale),
                int(img_right.shape[0] / self._scale),
            ),
        )

        self._image_right = img_right

        cv2.imshow("imgL", img_left)
        cv2.imshow("imgR", img_right)
        cv2.imshow("disparity", disparity)

        cv2.setMouseCallback("imgL", self.mouse_callback, "imgL")

        while True:
            key = cv2.waitKey(1)
            if key == 27:  # exit if ESC is pressed
                cv2.destroyAllWindows()
                break
