from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class MessageBoxController:
    def __init__(self):
        pass

    """
    Show error message in a jump-out window
    """

    @staticmethod
    def error_message_box(msg):
        ## generate pyqt5 message box
        message_box = QMessageBox()
        message_box.setWindowTitle("Error")
        message_box.setText(msg)
        message_box.setIcon(QMessageBox.Critical)
        x = message_box.exec_()
