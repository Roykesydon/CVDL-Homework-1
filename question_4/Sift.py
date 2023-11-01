import cv2
import numpy as np


class Sift:
    def __init__(self):
        pass

    def get_keypoints_and_descriptors(self, img: np.ndarray) -> tuple:
        # check if image is grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        return sift.detectAndCompute(img, None)

    def draw_keypoints(
        self, img: np.ndarray, key_points: list, color: tuple = (0, 0, 255)
    ):
        return cv2.drawKeypoints(img, key_points, None, color=color)

    def get_matched_image(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        # check if image is grayscale
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.get_keypoints_and_descriptors(img1)
        kp2, des2 = self.get_keypoints_and_descriptors(img2)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        img3 = cv2.drawMatchesKnn(
            img1,
            kp1,
            img2,
            kp2,
            good,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        return img3
