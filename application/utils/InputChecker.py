from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from application.controller.MessageBoxController import MessageBoxController

"""
Check GUI input before execute function
implemented by decorator
"""


class InputChecker:
    """
    check image list is empty or not
    """

    def check_image_list_empty(func):
        def wrapper(self):
            if len(self._image_loader.get_image_list()) == 0:
                MessageBoxController.error_message_box("Image list is empty.")
                return
            return func(self)

        return wrapper

    """
    check image left and right is empty or not
    """

    def check_image_left_and_right_empty(func):
        def wrapper(self):
            if self._image_loader.get_image_left_path() == None:
                MessageBoxController.error_message_box("Image left is empty.")
                return

            if self._image_loader.get_image_right_path() == None:
                MessageBoxController.error_message_box("Image right is empty.")
                return

            return func(self)

        return wrapper

    """
    check question 3's words input is legal or not
    """

    def check_words_is_legal(func):
        def wrapper(self):
            words = self._main_window.wordsText.toPlainText()
            # length must be 1~6
            if len(words) < 1 or len(words) > 6:
                MessageBoxController.error_message_box("Length of words must be 1~6.")
                return

            # words must be alphabets
            if not words.isalpha():
                MessageBoxController.error_message_box("Words must be alphabets.")
                return

            # change it to upper case
            self._words = words.upper()

            return func(self)

        return wrapper
