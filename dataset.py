from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import numpy as np


def load(filename):
    with open(filename, 'r') as f:
        return f.read().split("\n")


def crop_top(img, percent=0.15):
    offset = int(img.shape[0] * percent)
    return img[offset:]


def central_crop(img):
    size = min(img.shape[0], img.shape[1])
    offset_h = int((img.shape[0] - size) / 2)
    offset_w = int((img.shape[1] - size) / 2)
    return img[offset_h:offset_h + size, offset_w:offset_w + size]


class Dataset(data.Dataset):

    def __init__(self, pos_images: list[str], neg_images: list[str], transform: T.Compose = None):
        self.transform = transform
        self.pos_images = pos_images
        self.neg_images = neg_images
        self.all_images = []
        self.all_images.extend(pos_images)
        self.all_images.extend(neg_images)
        self.num_images = len(self.all_images)

    def __getitem__(self, index):

        file_path = self.all_images[index]
        label = 0 if file_path in self.pos_images else 1
        img = Image.open(file_path)
        img = np.array(img)
        if self.transform is not None:
            img = self.transform(Image.fromarray(img))
        return img, label

    def __len__(self):
        return self.num_images


def get_datasets():

    """Build and return a data loader."""
    train_pos = load("datasets-clean/covid/train_pos.txt")
    train_neg = load("datasets-clean/covid/train_neg.txt")
    test_pos = load("datasets-clean/covid/test_pos.txt")
    test_neg = load("datasets-clean/covid/test_neg.txt")

    test_transforms = [
        T.ToTensor(),
    ]

    train_transforms = [
        T.RandomHorizontalFlip()
    ]
    train_transforms.extend(test_transforms)

    train_transform = T.Compose(train_transforms)
    test_transform = T.Compose(test_transforms)

    train_dataset = Dataset(train_pos, train_neg, train_transform)
    test_dataset = Dataset(test_pos, test_neg, test_transform)

    return train_dataset, test_dataset
