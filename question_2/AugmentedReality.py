import cv2
import numpy as np

from utils.CvMatrixReader import CvMatrixReader


class AugmentedReality:
    def __init__(self):
        self.cv_matrix_reader = CvMatrixReader()
        self._alphabet_onboard_path = "./resource/alphabet_lib_onboard.txt"
        self._alphabet_vertical_path = "./resource/alphabet_lib_vertical.txt"

    """
    Draw lines on image
    """

    def draw(self, img, lines) -> np.ndarray:
        for point1, point2 in lines:
            color = (255, 0, 0)
            thickness = 8
            img = cv2.line(img, point1, point2, color, thickness)

        return img

    """
    According to the string to be displayed, and the camera related parameters
    Generate the coordinates of the line segment to be displayed on the image
    """

    def target_word_to_lines(
        self, word, calibrate_params_dict, img_index, vertical=False
    ) -> list:
        lines = []
        for idx, letter in enumerate(word):
            if not vertical:
                letter_matrix = self.cv_matrix_reader.read_file(
                    self._alphabet_onboard_path, letter
                )
            elif vertical:
                letter_matrix = self.cv_matrix_reader.read_file(
                    self._alphabet_vertical_path, letter
                )

            letter_matrix = np.reshape(letter_matrix, (-1, 3))

            letter_matrix[:, 1] += 2

            if idx < 3:
                letter_matrix[:, 1] += 3

            letter_matrix[:, 0] += 1 + 3 * (2 - idx % 3)

            img_points, jac = cv2.projectPoints(
                letter_matrix,
                calibrate_params_dict["rvecs"][img_index],
                calibrate_params_dict["tvecs"][img_index],
                calibrate_params_dict["instrinsic_matrix"],
                calibrate_params_dict["dist"],
            )
            img_points = np.reshape(img_points, (-1, 2)).astype(np.int32)

            lines += [
                (img_points[i], img_points[i + 1])
                for i in range(0, len(img_points) - 1, 2)
            ]

        return lines
