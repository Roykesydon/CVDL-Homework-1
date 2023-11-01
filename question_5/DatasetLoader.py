import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import v2

from .DataAugmentation import DataAugmentation


class DatasetLoader:
    def __init__(self, batch_size=4, data_augmentation_flag=False):
        self._batch_size = batch_size

        self._preprocessing = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self._transform = self._preprocessing

        if data_augmentation_flag:
            data_augmentation = DataAugmentation()
            self._transform = transforms.Compose(
                [data_augmentation.get_transforms(), self._preprocessing]
            )

        self._classes = (
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

    """
    load CIFAR10 dataset
    if dataset does not exist, download it
    clip_dataset_count: clip the train/test dataset to the first n data
    """

    def load_cifar10_dataset(self, clip_dataset_count=None):
        self._train_dataset = torchvision.datasets.CIFAR10(
            root="./dataset/CIFAR10",
            train=True,
            download=True,
            transform=self._transform,
        )

        if clip_dataset_count is not None:
            self._train_dataset = torch.utils.data.Subset(
                self._train_dataset, range(clip_dataset_count)
            )

        self._train_loader = torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=4,
        )

        self._test_dataset = torchvision.datasets.CIFAR10(
            root="./dataset/CIFAR10",
            train=False,
            download=True,
            transform=self._preprocessing,
        )

        if clip_dataset_count is not None:
            self._test_dataset = torch.utils.data.Subset(
                self._test_dataset, range(clip_dataset_count)
            )

        self._test_loader = torch.utils.data.DataLoader(
            self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=4,
        )

    """
    getters
    """

    def get_data_loader(self):
        return self._train_loader, self._test_loader

    def get_classes(self):
        return self._classes

    def get_batch_size(self):
        return self._batch_size

    # only do normalization
    def get_preprocessing(self):
        return self._preprocessing

    # only do data augmentation (without normalization)
    def get_transform(self):
        return self._transform
