# split into pos and neg train
from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import os
import numpy as np
from torchvision.utils import save_image

# dataset_dir = "datasets/covid"
#
# # preprocess training dataset
# pos_train = []
# neg_train = []
# with open(os.path.join(dataset_dir, "train.txt"), "r") as f:
#     for line in f:
#         x = line.strip().split(" ")
#         if x[2] == "positive":
#             pos_train.append(os.path.join(dataset_dir, "train", x[1]))
#         elif x[2] == "negative":
#             neg_train.append(os.path.join(dataset_dir, "train", x[1]))
#         else:
#             print(x)
#
# with open(os.path.join(dataset_dir, "train_pos.txt"), "w") as f:
#     f.write("\n".join(pos_train))
# with open(os.path.join(dataset_dir, "train_neg.txt"), "w") as f:
#     f.write("\n".join(neg_train))
#
# # preprocess test dataset
# pos_test = []
# neg_test = []
# with open(os.path.join(dataset_dir, "test.txt"), "r") as f:
#     for line in f:
#         x = line.strip().split(" ")
#         if x[2] == "positive":
#             pos_test.append(os.path.join(dataset_dir, "test", x[1]))
#         elif x[2] == "negative":
#             neg_test.append(os.path.join(dataset_dir, "test", x[1]))
#         else:
#             print(x)
#
# with open(os.path.join(dataset_dir, "test_pos.txt"), "w") as f:
#     f.write("\n".join(pos_test))
# with open(os.path.join(dataset_dir, "test_neg.txt"), "w") as f:
#     f.write("\n".join(neg_test))


def extract_file_label(source_file, target_dir, save_pos_file, save_neg_file):
    pos = []
    neg = []
    with open(source_file, "r") as f:
        for line in f:
            x = line.strip().split(" ")
            if x[2] == "positive":
                pos.append(x[1])
            elif x[2] == "negative":
                neg.append(x[1])
            else:
                print(x)

    with open(save_pos_file, "w") as f:
        f.write("\n".join([os.path.join(target_dir, filename) for filename in pos]))
    with open(save_neg_file, "w") as f:
        f.write("\n".join([os.path.join(target_dir, filename) for filename in neg]))
    return pos, neg


train_source_file = "datasets/covid/train.txt"
test_source_file = "datasets/covid/test.txt"
train_pos_file = "datasets-clean/covid/train_pos.txt"
train_neg_file = "datasets-clean/covid/train_neg.txt"
test_pos_file = "datasets-clean/covid/test_pos.txt"
test_neg_file = "datasets-clean/covid/test_neg.txt"
train_target_dir = "datasets-clean/covid/train"
test_target_dir = "datasets-clean/covid/test"

pos_train, neg_train = extract_file_label(train_source_file, train_target_dir, train_pos_file, train_neg_file)
pos_test, neg_test = extract_file_label(test_source_file, test_target_dir, test_pos_file, test_neg_file)


def crop_top(img, percent=0.15):
    offset = int(img.shape[0] * percent)
    return img[offset:]


def central_crop(img):
    size = min(img.shape[0], img.shape[1])
    offset_h = int((img.shape[0] - size) / 2)
    offset_w = int((img.shape[1] - size) / 2)
    return img[offset_h:offset_h + size, offset_w:offset_w + size]


transforms = [
    T.Resize(256),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Standardize to [-1, 1]
]

# train_transforms = [
#     T.RandomHorizontalFlip(),
# ]
# train_transforms.extend(transforms)

# train_transform = T.Compose(train_transforms)
# test_transform = T.Compose(transforms)
transform = T.Compose(transforms)

# def preprocess_images(img_paths, img_full_paths, save_dir):
#     for idx, file_path in enumerate(img_full_paths):
#         save_path = os.path.join(save_dir, img_paths[idx])
#         print(f"Processing {file_path} ==> {save_path}")
#
#         img = Image.open(file_path)
#         img = img.convert("RGB")
#         img = np.array(img)
#         img = crop_top(img, 0.08)
#         img = central_crop(img)
#         img = transform(Image.fromarray(img))
#         save_image(img, save_path)


def preprocess_images(source_dir: str, target_dir: str, img_files: list[str]):
    for img_file in img_files:
        open_path = os.path.join(source_dir, img_file)
        save_path = os.path.join(target_dir, img_file)
        print(f"Processing {open_path} ==> {save_path}")

        img = Image.open(open_path)
        img = img.convert("RGB")
        img = np.array(img)
        img = crop_top(img, 0.08)
        img = central_crop(img)
        img = transform(Image.fromarray(img))
        save_image(img, save_path)


cleaned_pos_train_dir = "datasets-clean/covid/train/pos"
cleaned_neg_train_dir = "datasets-clean/covid/train/neg"
cleaned_pos_test_dir = "datasets-clean/covid/test/pos"
cleaned_neg_test_dir = "datasets-clean/covid/test/neg"

# preprocess_images(pos_train, pos_train_full_path, cleaned_pos_train_dir)
# preprocess_images(neg_train, neg_train_full_path, cleaned_neg_train_dir)
# preprocess_images(pos_test, pos_test_full_path, cleaned_pos_test_dir)
# preprocess_images(neg_test, neg_test_full_path, cleaned_neg_test_dir)

train_source_dir = "datasets/covid/train"
test_source_dir = "datasets/covid/test"
preprocess_images(train_source_dir, cleaned_pos_train_dir, pos_train)
preprocess_images(train_source_dir, cleaned_neg_train_dir, neg_train)
preprocess_images(test_source_dir, cleaned_pos_test_dir, pos_test)
preprocess_images(test_source_dir, cleaned_neg_test_dir, neg_test)

# for file_path in pos_train:
#     save_path = os.path.join(cleaned_pos_train_dir, file_path)
#     img = Image.open(file_path)
#     img = img.convert("RGB")
#     img = np.array(img)
#     img = crop_top(img, 0.08)
#     img = central_crop(img)
#     img = train_transform(Image.fromarray(img))
#     save_image(img)
#
# for file_path in neg_train:
#     save_path = os.path.join(cleaned_neg_train_dir, file_path)
#     img = Image.open(file_path)
#     img = img.convert("RGB")
#     img = np.array(img)
#     img = crop_top(img, 0.08)
#     img = central_crop(img)
#     img = train_transform(Image.fromarray(img))
#     save_image(img)
#
# for file_path in pos_test:
#     save_path = os.path.join(cleaned_pos_test_dir, file_path)
#     img = Image.open(file_path)
#     img = img.convert("RGB")
#     img = np.array(img)
#     img = crop_top(img, 0.08)
#     img = central_crop(img)
#     img = train_transform(Image.fromarray(img))
#     save_image(img)
#
# for file_path in neg_test:
#     save_path = os.path.join(cleaned_neg_test_dir, file_path)
#     img = Image.open(file_path)
#     img = img.convert("RGB")
#     img = np.array(img)
#     img = crop_top(img, 0.08)
#     img = central_crop(img)
#     img = train_transform(Image.fromarray(img))
#     save_image(img)