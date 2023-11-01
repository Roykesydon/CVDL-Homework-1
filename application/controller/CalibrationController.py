import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from application.utils.InputChecker import InputChecker
from question_1.CameraCalibration import CameraCalibration
from utils.PrintFormatter import PrintFormatter

from .DisplayMultiImageController import DisplayMultiImageController


class CalibrationController:
    def __init__(self, main_window, image_loader):
        self._main_window = main_window
        self._image_loader = image_loader

        self._instrinsic_matrix = None
        self._rvecs = None
        self._tvecs = None
        self._dist = None

        self._camera_calibration = CameraCalibration()
        self._print_formatter = PrintFormatter()

        self._display_multi_image_controller = None

        self._main_window.findCorners.clicked.connect(self.on_findCorners_clicked)
        self._main_window.findIntrinsic.clicked.connect(self.on_findIntrinsic_clicked)
        self._main_window.findExtrinsic.clicked.connect(self.on_findExtrinsic_clicked)
        self._main_window.findDistortion.clicked.connect(self.on_findDistortion_clicked)
        self._main_window.showResult.clicked.connect(self.on_showResult_clicked)

    @InputChecker.check_image_list_empty
    def on_findCorners_clicked(self):
        def draw_chess_board_corner(img_path: str):
            img = cv2.imread(img_path)

            width, height = 11, 8

            ret, corners = self._camera_calibration.find_chessboard_corner(
                img, width, height
            )
            if ret == True:
                # draw chessboard corner at image
                img = self._camera_calibration.draw_chess_board_corner(
                    img, width, height, corners
                )
            else:
                print("Chessboard corner not found.")

            img = cv2.resize(img, (480, 480))

            return img

        self._display_multi_image_controller = DisplayMultiImageController()
        self._display_multi_image_controller.set_image_loader(self._image_loader)
        self._display_multi_image_controller.set_process_function(
            draw_chess_board_corner
        )
        self._display_multi_image_controller.start_display()

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

    @InputChecker.check_image_list_empty
    def on_findIntrinsic_clicked(self):
        if self._instrinsic_matrix is None:
            self.calibrate_with_all_images()

        # print instrinsic matrix
        self._print_formatter.print_matrix(self._instrinsic_matrix, "Intrinsic Matrix")

    @InputChecker.check_image_list_empty
    def on_findExtrinsic_clicked(self):
        extrinsic_number = self._main_window.extrinsicNumber.currentText()

        if self._rvecs is None or self._tvecs is None:
            self.calibrate_with_all_images()

        rvec = self._rvecs[int(extrinsic_number) - 1]
        tvec = self._tvecs[int(extrinsic_number) - 1]

        extrinsic_matrix = self._camera_calibration.get_extrinsic_matrix(rvec, tvec)

        # print extrinsic matrix
        # ppt's example is 11.bmp's extrinsic matrix
        self._print_formatter.print_matrix(extrinsic_matrix, "Extrinsic Matrix")

    @InputChecker.check_image_list_empty
    def on_findDistortion_clicked(self):
        if self._dist is None:
            self.calibrate_with_all_images()

        # print distortion matrix
        self._print_formatter.print_matrix(self._dist, "Distortion Matrix")

    @InputChecker.check_image_list_empty
    def on_showResult_clicked(self):
        if self._instrinsic_matrix is None or self._dist is None:
            self.calibrate_with_all_images()

        def undistort_image(img_path: str):
            img = cv2.imread(img_path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            undistort_img = self._camera_calibration.undistort_image(
                img, self._instrinsic_matrix, self._dist
            )

            # make img_gray have same size as undistort_img
            img_gray = cv2.resize(
                img_gray, (undistort_img.shape[1], undistort_img.shape[0])
            )

            # merge them to one image
            merge_img = np.hstack((img_gray, undistort_img))

            merge_img = cv2.resize(merge_img, (480 * 2, 480))

            return merge_img

        self._display_multi_image_controller = DisplayMultiImageController(
            width=480 * 2, height=480
        )
        self._display_multi_image_controller.set_image_loader(self._image_loader)
        self._display_multi_image_controller.set_process_function(undistort_image)
        self._display_multi_image_controller.start_display()
