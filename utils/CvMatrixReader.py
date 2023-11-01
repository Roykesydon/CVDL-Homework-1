import cv2
import numpy as np


class CvMatrixReader:
    def __init__(self) -> None:
        pass

    """
    Read matrix related to letter from given file
    """

    def read_file(self, file_path, letter) -> np.ndarray:
        file_storage = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
        return file_storage.getNode(letter).mat().astype(np.float32)
