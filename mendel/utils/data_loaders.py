import enum
from typing import List

import numpy as np
import pydantic
from torch.utils.data import Dataset

# from torchtext.datasets import CoLA
from torchvision.datasets import CIFAR10, MNIST, Omniglot


class MnistClass(int, enum.Enum):
    Zero = 0
    One = 1
    Two = 2
    Three = 3
    Four = 4
    Five = 5
    Six = 6
    Seven = 7
    Eight = 8
    Nine = 9


class CifarTenClass(int, enum.Enum):
    Plane = 0
    Car = 1
    Bird = 2
    Cat = 3
    Deer = 4
    Dog = 5
    Frog = 6
    Horse = 7
    Ship = 8
    Truck = 9


class MaskingMode(int, enum.Enum):
    # Create a dataset that only has a specified class
    Include = 1

    # Create a dataset that does not have a specified class
    Exclude = 2

    # Create a dataset with all the classes
    NoMasking = 3


class DataMode(str, enum.Enum):
    Test = "Test"
    Train = "Train"


def include_single_class(
    data: np.array, labels: pydantic.conlist(item_type=int), class_id: int
) -> List:
    """
    Extracts, from the dataset `data`, all the data belonging to the ith class
    """
    labels = np.array(labels)
    locations_of_class = np.argwhere(labels == class_id)
    locations_of_class = list(locations_of_class[:, 0])

    return [data[i] for i in locations_of_class]


def exclude_single_class(
    data: np.array, labels: pydantic.conlist(item_type=int), class_id: int
) -> List:
    """
    Extracts, from the dataset `data`, all the data that does NOT belong
    to the ith class
    """
    labels = np.array(labels)
    locations_of_class = np.argwhere(labels != class_id)
    locations_of_class = list(locations_of_class[:, 0])
    return [data[i] for i in locations_of_class]


class MaskedMnistDataset(Dataset):
    def __init__(
        self, data_mode: DataMode, masking_mode: MaskingMode, masking_class: MnistClass
    ) -> None:
        if data_mode == DataMode.Train:
            dataset = MNIST(root="./data", train=True, download=True)
        else:
            dataset = MNIST(root="./data", train=False, download=True)
        self.datasets = dataset
        self.lengths = None


class MaskedCifarTenDataset(Dataset):
    def __init__(
        self, data_mode: DataMode, masking_mode: MaskingMode, masking_class: MnistClass
    ) -> None:
        if data_mode == DataMode.Train:
            dataset = CIFAR10(root="./data", train=True, download=True)
        else:
            dataset = CIFAR10(root="./data", train=False, download=True)
        self.datasets = dataset


class MaskedOmniglotDataset(Dataset):
    def __init__(
        self, data_mode: DataMode, masking_mode: MaskingMode, masking_class: MnistClass
    ) -> None:
        if data_mode == DataMode.Train:
            dataset = Omniglot(root="./data", train=True, download=True)
        else:
            dataset = Omniglot(root="./data", train=False, download=True)
        self.datasets = dataset
