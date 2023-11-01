import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from torchvision import transforms

from question_5.DataAugmentation import DataAugmentation
from question_5.DatasetLoader import DatasetLoader
from question_5.VggModel import VggModel
from utils.ImageLoader import ImageLoader

from .MessageBoxController import MessageBoxController


class VggController:
    def __init__(self, main_window):
        self.RESULT_CHART_PATH = "./resource/result_final.png"
        self.MODEL_PATH = "./weights/vgg19_bn_final.pth"
        self.Q5_1_DATASET_PATH = "./resource/Q5_1"

        self._main_window = main_window
        self._vgg_model = VggModel()
        self._vgg_model.load_model(model_path=self.MODEL_PATH)

        self._image_loader = ImageLoader()
        self._data_augmentation = DataAugmentation()
        self._dataset_loader = DatasetLoader(
            batch_size=128, data_augmentation_flag=True
        )

        self._inference_img_path = None

        self._main_window.loadImage.clicked.connect(self.on_loadImage_clicked)
        self._main_window.showAugmentedImages.clicked.connect(
            self.on_showAugmentedImages_clicked
        )
        self._main_window.showModelStructure.clicked.connect(
            self.on_showModelStructure_clicked
        )
        self._main_window.showAccuracyAndLoss.clicked.connect(
            self.on_showAccuracyAndLoss_clicked
        )
        self._main_window.inference.clicked.connect(self.on_inference_clicked)

    def on_loadImage_clicked(self):
        self._inference_img_path = str(
            QFileDialog.getOpenFileName(None, "Select Image")[0]
        )
        # Display image on QLabel (name: inferenceImage)
        self._main_window.inferenceImage.setPixmap(QPixmap(self._inference_img_path))
        self._main_window.inferenceImage.setScaledContents(True)

    def on_showAugmentedImages_clicked(self):
        # Load images
        self._image_loader.load_from_folder(self.Q5_1_DATASET_PATH)

        # Get images
        img_path_list = self._image_loader.get_all_images_with_abs_path()

        # Load images as numpy array
        img_list = list(map(lambda x: Image.open(x), img_path_list))

        # Data augmentation
        img_list = list(
            map(
                lambda x: transforms.Compose(
                    [
                        self._data_augmentation.get_transforms(),
                        transforms.ToTensor(),
                    ]
                )(x),
                img_list,
            )
        )

        # Display images as a 3x3 grid in a single image
        # show it with correct RGB and filename
        fig, axs = plt.subplots(3, 3)
        fig.tight_layout()
        for i in range(3):
            for j in range(3):
                img = img_list[i * 3 + j].numpy()
                img = np.transpose(img, (1, 2, 0))

                axs[i, j].imshow(img)
                axs[i, j].set_title(
                    os.path.basename(img_path_list[i * 3 + j]).split(".")[0]
                )

        plt.show()

    def on_showModelStructure_clicked(self):
        self._vgg_model.print_model_summary()

    def on_showAccuracyAndLoss_clicked(self):
        # load img and show
        img = cv2.imread(self.RESULT_CHART_PATH)
        cv2.imshow("result chart", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_inference_clicked(self):
        if self._inference_img_path is None:
            MessageBoxController.error_message_box("Please select an image first!")
            return

        outputs = self._vgg_model.inference(
            self._inference_img_path,
            self._dataset_loader.get_preprocessing(),
        )

        self._vgg_model.show_inference_probability(
            outputs, self._dataset_loader.get_classes()
        )

        predict_class = self._dataset_loader.get_classes()[outputs.argmax()]
        self._main_window.predictText.setText(predict_class)
