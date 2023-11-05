import os

import cv2
import numpy as np


class ImageLoader:
    def __init__(self):
        self._image_folder_path = ""
        self._image_list = []
        self._image_index = 0

        self._image_left_path = None
        self._image_right_path = None

    """
    Load image from given folder path
    """

    def load_from_folder(self, folder_path: str) -> None:
        # check if folder exists
        if not os.path.exists(folder_path):
            print(f"{folder_path} does not exist.")
            return

        self._image_folder_path = folder_path
        self._image_list = os.listdir(folder_path)

        # check if folder_path is absolute path
        if not os.path.isabs(folder_path):
            folder_path = os.path.abspath(folder_path)

        # filter out folder
        self._image_list = list(
            filter(
                lambda x: not os.path.isdir(folder_path + os.sep + x), self._image_list
            )
        )

        # sort image by name
        # if image name is 1.png, 2.png, 3.png, 4.png, 5.png
        # check if image name is number
        if self._image_list[0].split("/")[-1].split(".")[0].isdigit():
            self._image_list.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
        else:
            self._image_list.sort()

        self._image_index = 0

    """
    return image as numpy array
    """

    def load_from_image_path(self, image_path: str, gray: bool = False) -> np.ndarray:
        # check if image exists
        if not os.path.exists(image_path):
            print(f"{image_path} does not exist.")
            return

        if gray:
            return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        return cv2.imread(image_path)

    def get_all_images_with_abs_path(self) -> list:
        folder_path = self._image_folder_path

        # check if folder_path is absolute path
        if not os.path.isabs(folder_path):
            folder_path = os.path.abspath(folder_path)

        return list(map(lambda x: folder_path + os.sep + x, self._image_list))

    def get_current_image(self) -> str:
        folder_path = self._image_folder_path

        if len(self._image_list) == 0:
            print("Image list is empty.")
            return None

        # check if folder_path is absolute path
        if not os.path.isabs(folder_path):
            folder_path = os.path.abspath(folder_path)

        return folder_path + os.sep + self._image_list[self._image_index]

    """
    dummy getter and setter
    """

    def set_image_left_path(self, image_path: str) -> None:
        self._image_left_path = image_path

    def set_image_right_path(self, image_path: str) -> None:
        self._image_right_path = image_path

    def get_image_list(self) -> list:
        return self._image_list

    def get_folder_path(self) -> str:
        return self._image_folder_path

    def get_image_index(self) -> int:
        return self._image_index

    def get_image_left_path(self) -> str:
        return self._image_left_path

    def get_image_right_path(self) -> str:
        return self._image_right_path

    """
    Change current image index by given step
    """

    def image_index_step(self, step: int = 1) -> None:
        self._image_index += step
        if self._image_index >= len(self._image_list):
            self._image_index %= len(self._image_list)


if __name__ == "__main__":
    img_loader = ImageLoader()
    img_loader.load_from_folder("./dataset/Dataset_CvDl_Hw1/Q1_Image")
