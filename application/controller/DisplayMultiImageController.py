import cv2
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class DisplayMultiImageController(QDialog):
    def __init__(self, width=480, height=480):
        super().__init__()
        self.setWindowTitle("Image Display")
        self.setGeometry(100, 100, width, height)

        self._kwargs = {}

        layout = QVBoxLayout()
        self.label = QLabel(self)
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.current_image_index = 0
        self.image_loader = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.load_next_image)
        self.show()

    def set_image_loader(self, image_loader):
        self.image_loader = image_loader

    """
    process image before display
    """

    def set_process_function(self, process_function):
        self._process_function = process_function

    """
    set kwargs for process function
    """

    def set_kwargs(self, **kwargs):
        self._kwargs = kwargs

    def load_next_image(self):
        if self.image_loader is None:
            return

        img_path = self.image_loader.get_current_image()
        if img_path is None:
            return

        if self._process_function is not None:
            img = self._process_function(img_path, **self._kwargs)
        else:
            img = cv2.imread(img_path)

        self.display_image(img)
        self.image_loader.image_index_step(1)

    def display_image(self, img):
        if img is not None:
            if len(img.shape) == 2:
                height, width = img.shape
                bytes_per_line = width
                q_image = QImage(
                    img.data, width, height, bytes_per_line, QImage.Format_Grayscale8
                )
            elif len(img.shape) == 3:
                height, width, channel = img.shape
                bytes_per_line = 3 * width
                q_image = QImage(
                    img.data, width, height, bytes_per_line, QImage.Format_RGB888
                )
            pixmap = QPixmap.fromImage(q_image)
            self.label.setPixmap(pixmap)

    def start_display(self):
        self.timer.start(1000)

    """
    Override close event
    """

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()
