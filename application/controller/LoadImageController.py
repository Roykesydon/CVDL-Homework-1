import sys

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class LoadImageController:
    def __init__(self, main_window, image_loader):
        self._main_window = main_window
        self._image_loader = image_loader

        self._main_window.loadFolder.clicked.connect(self.on_loadFolder_clicked)
        self._main_window.loadImageL.clicked.connect(self.on_loadImageL_clicked)
        self._main_window.loadImageR.clicked.connect(self.on_loadImageR_clicked)

    """
    Load image folder for question 1 and 2
    """

    def on_loadFolder_clicked(self):
        folder_path = str(QFileDialog.getExistingDirectory(None, "Select Directory"))
        print(folder_path)
        self._image_loader.load_from_folder(folder_path)

    """
    Load image left for question 3
    """

    def on_loadImageL_clicked(self):
        self._image_loader.set_image_left_path(
            str(QFileDialog.getOpenFileName(None, "Select Image")[0])
        )

    """
    Load image right for question 3
    """

    def on_loadImageR_clicked(self):
        self._image_loader.set_image_right_path(
            str(QFileDialog.getOpenFileName(None, "Select Image")[0])
        )
