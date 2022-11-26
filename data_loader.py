from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import os
import numpy as np

# datasetA: positive --> generator input
# datasetB: negative --> discriminator input
dataset_dir = "datasets/covid"


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

    def __init__(self, datasetA, datasetB):
        transform = [
            T.RandomHorizontalFlip(),
            T.Resize(256),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # Standardize to [-1, 1]
        ]

        self.transform = T.Compose(transform)
        self.datasetA = datasetA
        self.datasetB = datasetB
        self.num_images = len(self.datasetA) + len(self.datasetB)

    def __getitem__(self, index):
        datasetA = self.datasetA
        datasetB = self.datasetB

        filenameA = datasetA[index % len(datasetA)]
        filenameB = datasetB[index % len(datasetB)]

        imageA = Image.open(filenameA).convert("RGB")
        imageB = Image.open(filenameB).convert("RGB")

        imageA = np.array(imageA)
        imageA = crop_top(imageA, 0.08)
        imageA = central_crop(imageA)

        imageB = np.array(imageB)
        imageB = crop_top(imageB, 0.08)
        imageB = central_crop(imageB)

        imageA = self.transform(Image.fromarray(imageA))
        imageB = self.transform(Image.fromarray(imageB))

        return imageA, imageB

    def __len__(self):
        return self.num_images


class TestDataset(data.Dataset):
    """Dataset class for the Covid datasets."""

    def __init__(self, dataset):
        """Initialize and preprocess the Covid datasets."""
        transform = [
            T.Resize(256),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Standardize to [-1, 1]
        ]
        self.transform = T.Compose(transform)
        self.dataset = dataset
        self.num_images = len(dataset)

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.dataset

        filename = dataset[index % len(dataset)]
        image = Image.open(os.path.join(filename)).convert("RGB")
        image = np.array(image)
        image = crop_top(image, 0.08)
        image = central_crop(image)
        image = Image.fromarray(image)

        return self.transform(image)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader():
    """Build and return a data loader."""
    train_pos = load(os.path.join(dataset_dir, "train_pos.txt"))
    train_neg = load(os.path.join(dataset_dir, "train_neg.txt"))
    train_dataset = Dataset(train_pos, train_neg)
    data_loader = data.DataLoader(dataset=train_dataset,
                                  batch_size=16,
                                  shuffle=True,
                                  num_workers=12)
    return data_loader


def get_pos_test_loader():
    test_pos = load(os.path.join(dataset_dir, "test_pos.txt"))
    test_anomalous = TestDataset(test_pos)
    data_loader = data.DataLoader(dataset=test_anomalous,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=12)
    return data_loader


def get_neg_test_loader():
    test_neg = load(os.path.join(dataset_dir, "test_neg.txt"))
    test_healthy = TestDataset(test_neg)
    data_loader = data.DataLoader(dataset=test_healthy,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=12)
    return data_loader


# class Covid(data.Dataset):
#     """Dataset class for the Covid datasets."""
#
#     def __init__(self, image_dir, transform, mode):
#         """Initialize and preprocess the Covid datasets."""
#         self.image_dir = image_dir
#         self.transform = transform
#         self.mode = mode
#         self.datasetA = []
#         self.datasetB = []
#         self.preprocess()
#
#         if mode == 'train':
#             self.num_images = len(self.datasetA) + len(self.datasetB)
#         else:
#             self.num_images = max(len(self.datasetA), len(self.datasetB))
#
#     def preprocess(self):
#         if self.mode in ['train', 'test2']:
#             pos = load(os.path.join("datasets", "covid", "train_pos.txt"))
#             neg = load(os.path.join("datasets", "covid", "train_neg.txt"))
#             # neg_mixed = load(os.path.join("data", "covid", "train_neg_mixed"))
#
#             # self.datasetA = pos + neg_mixed
#             self.datasetA = pos
#             self.datasetB = neg
#         else:
#             self.datasetA = load(os.path.join("datasets", "covid", "test_pos.txt"))
#             self.datasetB = load(os.path.join("datasets", "covid", "test_neg.txt"))
#
#         print('Finished preprocessing the COVID datasets...')
#
#     def __getitem__(self, index):
#         """Return one image and its corresponding attribute label."""
#         datasetA = self.datasetA
#         datasetB = self.datasetB
#
#         filenameA = datasetA[index % len(datasetA)]
#         filenameB = datasetB[index % len(datasetB)]
#
#         if self.mode in ['train']:
#             imageA = Image.open(os.path.join(self.image_dir, 'train', filenameA)).convert("RGB")
#             imageB = Image.open(os.path.join(self.image_dir, 'train', filenameB)).convert("RGB")
#         else:
#             imageA = Image.open(os.path.join(self.image_dir, 'test', filenameA)).convert("RGB")
#             imageB = Image.open(os.path.join(self.image_dir, 'test', filenameB)).convert("RGB")
#
#         imageA = np.array(imageA)
#         imageA = crop_top(imageA, 0.08)
#         imageA = central_crop(imageA)
#
#         imageB = np.array(imageB)
#         imageB = crop_top(imageB, 0.08)
#         imageB = central_crop(imageB)
#
#         imageA = Image.fromarray(imageA)
#         imageB = Image.fromarray(imageB)
#
#         return self.transform(imageA), self.transform(imageB)
#
#     def __len__(self):
#         """Return the number of images."""
#         return self.num_images
#
#
# class TestValid(data.Dataset):
#     """Dataset class for the Covid datasets."""
#
#     def __init__(self, image_dir, transform, mode):
#         """Initialize and preprocess the Covid datasets."""
#         self.image_dir = image_dir
#         self.transform = transform
#         self.mode = mode
#         self.datasetA = []
#         self.datasetB = []
#         self.preprocess()
#
#         if "ano" in self.mode:
#             self.num_images = len(self.datasetA)
#         elif "hea" in self.mode:
#             self.num_images = len(self.datasetB)
#
#     def preprocess(self):
#         self.datasetA = load(os.path.join("datasets", "covid", "test_pos.txt"))
#         self.datasetB = load(os.path.join("datasets", "covid", "test_neg.txt"))
#
#         print(f'Finished preprocessing the COVID datasets for {self.mode} ...')
#
#     def __getitem__(self, index):
#         """Return one image and its corresponding attribute label."""
#         if "ano" in self.mode:
#             dataset = self.datasetA
#         else:
#             dataset = self.datasetB
#
#         filename = dataset[index%len(dataset)]
#         image = Image.open(os.path.join(self.image_dir, 'test', filename)).convert("RGB")
#
#         image = np.array(image)
#         image = crop_top(image, 0.08)
#         image = central_crop(image)
#
#         image = Image.fromarray(image)
#
#         return self.transform(image)
#
#     def __len__(self):
#         """Return the number of images."""
#         return self.num_images
