import sys

from PyQt5 import QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from application.controller.AugmentedRealityController import AugmentedRealityController
from application.controller.CalibrationController import CalibrationController
from application.controller.LoadImageController import LoadImageController
from application.controller.SiftController import SiftController
from application.controller.StereoDisparityController import StereoDisparityController
from application.controller.VggController import VggController
from application.xml import main_ui
from utils.ImageLoader import ImageLoader


class myMainWindow(QMainWindow, main_ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self._image_loader = ImageLoader()

        self._loadImageController = LoadImageController(self, self._image_loader)
        self._calibrationController = CalibrationController(self, self._image_loader)
        self._augmentedRealityController = AugmentedRealityController(
            self, self._image_loader
        )
        self._stereoDisparityController = StereoDisparityController(
            self, self._image_loader
        )
        self._siftController = SiftController(self, self._image_loader)
        self._vggController = VggController(self)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = myMainWindow()
    window.show()
    sys.exit(app.exec_())
