import cv2
import numpy as np


class CameraCalibration:
    """
    Detect chessboard corner
    """

    def find_chessboard_corner(self, img, width, height) -> (bool, np.ndarray):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray_img, (width, height), None)

        win_size = (5, 5)
        zero_zone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        if ret == True:
            # find more accurate corner
            corners = cv2.cornerSubPix(gray_img, corners, win_size, zero_zone, criteria)

        return ret, corners

    """
    Draw chessboard corner at image
    """

    def draw_chess_board_corner(self, img, width, height, corners) -> np.ndarray:
        img = cv2.drawChessboardCorners(img.copy(), (width, height), corners, True)

        return img

    """
    Get camera calibration parameters
    """

    def calibrate_camera(
        self, img, width, height, img_points
    ) -> (bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        object_point = np.zeros((height * width, 3), np.float32)
        object_point[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

        obj_points = [object_point for _ in range(len(img_points))]

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, gray_img.shape[::-1], None, None
        )

        return ret, mtx, dist, rvecs, tvecs

    """
    Transform rotation vector to rotation matrix
    And then combine with translation vector to extrinsic matrix
    """

    def get_extrinsic_matrix(self, rvec, tvec) -> np.ndarray:
        # transform rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        extrinsic_matrix = np.hstack((rmat, tvec))

        return extrinsic_matrix

    """
    Undistort image with camera calibration parameters
    """

    def undistort_image(self, img, mtx, dist) -> np.ndarray:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray_img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        dst = cv2.undistort(gray_img, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y : y + h, x : x + w]

        return dst
