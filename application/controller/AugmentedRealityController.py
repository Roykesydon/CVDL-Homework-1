import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from application.utils.InputChecker import InputChecker
from question_1.CameraCalibration import CameraCalibration
from question_2.AugmentedReality import AugmentedReality

from .DisplayMultiImageController import DisplayMultiImageController


class AugmentedRealityController:
    def __init__(self, main_window, image_loader):
        self._main_window = main_window

        self._image_loader = image_loader
        self._augmented_reality = AugmentedReality()
        self._camera_calibration = CameraCalibration()

        self._words = ""

        self._instrinsic_matrix = None
        self._rvecs = None
        self._tvecs = None
        self._dist = None

        self._main_window.showWordsHorizontal.clicked.connect(
            self.on_showWordsHorizontal_clicked
        )
        self._main_window.showWordsVertical.clicked.connect(
            self.on_showWordsVertical_clicked
        )

    @InputChecker.check_image_list_empty
    def calibrate_with_all_images(self):
        img_points = []

        for img_path in self._image_loader.get_all_images_with_abs_path():
            img = cv2.imread(img_path)

            width, height = 11, 8

            ret, corners = self._camera_calibration.find_chessboard_corner(
                img, width, height
            )
            if ret == True:
                img_points.append(corners)
            else:
                print("Chessboard corner not found.")

        # get camera calibration parameters
        (
            _,
            instrinsic_matrix,
            dist,
            rvecs,
            tvecs,
        ) = self._camera_calibration.calibrate_camera(img, width, height, img_points)

        self._instrinsic_matrix = instrinsic_matrix
        self._dist = dist
        self._rvecs = rvecs
        self._tvecs = tvecs

    def show_on_board(img_path: str, **kwargs):
        img = cv2.imread(img_path)

        self = kwargs["augmented_reality_controller"]
        vertical = kwargs["vertical"]

        calibrate_params_dict = {
            "instrinsic_matrix": self._instrinsic_matrix,
            "dist": self._dist,
            "rvecs": self._rvecs,
            "tvecs": self._tvecs,
        }

        # get lines according to target word
        lines = self._augmented_reality.target_word_to_lines(
            self._words,
            calibrate_params_dict,
            self._image_loader.get_image_index(),
            vertical=vertical,
        )

        img = self._augmented_reality.draw(img, lines)
        img = cv2.resize(img, (480, 480))

        return img

    @InputChecker.check_image_list_empty
    @InputChecker.check_words_is_legal
    def on_showWordsHorizontal_clicked(self):
        if self._instrinsic_matrix is None or self._dist is None:
            self.calibrate_with_all_images()

        self._display_multi_image_controller = DisplayMultiImageController(
            width=480, height=480
        )
        self._display_multi_image_controller.set_image_loader(self._image_loader)
        self._display_multi_image_controller.set_process_function(
            AugmentedRealityController.show_on_board
        )
        self._display_multi_image_controller.set_kwargs(
            augmented_reality_controller=self, vertical=False
        )
        self._display_multi_image_controller.start_display()

    @InputChecker.check_image_list_empty
    @InputChecker.check_words_is_legal
    def on_showWordsVertical_clicked(self):
        if self._instrinsic_matrix is None or self._dist is None:
            self.calibrate_with_all_images()

        self._display_multi_image_controller = DisplayMultiImageController(
            width=480, height=480
        )
        self._display_multi_image_controller.set_image_loader(self._image_loader)
        self._display_multi_image_controller.set_process_function(
            AugmentedRealityController.show_on_board
        )
        self._display_multi_image_controller.set_kwargs(
            augmented_reality_controller=self, vertical=True
        )
        self._display_multi_image_controller.start_display()
