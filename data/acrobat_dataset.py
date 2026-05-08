import os
from PIL import Image
import numpy as np
import random
import torchvision.transforms as transforms
from PIL import ImageFile
from data.base_dataset import BaseDataset
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd


class AcrobatDataset(BaseDataset):
    def __init__(self, opt, train=True):
        BaseDataset.__init__(self, opt)
        default_dataroot = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "acrobat_dataset",
                "processed_databag",
            )
        )
        self.train = True if (train or opt.phase == "train") else False
        self.dataroot = (
            opt.dataroot
            if opt.dataroot not in (None, "placeholder")
            else default_dataroot
        )
        self.csv_path = (
            opt.csv_path
            if opt.csv_path not in (None, "placeholder")
            else os.path.join(self.dataroot, "shutiled_patches", "split.csv")
        )
        self.stain = opt.stain
        self.load_size = opt.load_size
        self.crop_size = opt.crop_size
        self.no_flip = opt.no_flip
        self.misalign = getattr(opt, "misalign", 0)
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        if self.load_size == 256:
            df_train = df[
                (df["patch_type"] == "subpatch_256") & (df["split"] == "train")
            ]
            df_test = df[(df["patch_type"] == "subpatch_256") & (df["split"] == "test")]
            train_paths = [
                os.path.join(
                    self.dataroot,
                    row["slide_path"],
                    "subpatch_256",
                    "HE",
                    row["file_path"],
                )
                for _, row in df_train.iterrows()
            ]
            test_paths = [
                os.path.join(
                    self.dataroot,
                    row["slide_path"],
                    "subpatch_256",
                    "HE",
                    row["file_path"],
                )
                for _, row in df_test.iterrows()
            ]
        elif self.load_size == 512:
            df_train = df[(df["patch_type"] == "patch_512") & (df["split"] == "train")]
            df_test = df[(df["patch_type"] == "patch_512") & (df["split"] == "test")]
            train_paths = [
                os.path.join(
                    self.dataroot,
                    row["slide_path"],
                    "patch_512",
                    "HE",
                    row["file_path"],
                )
                for _, row in df_train.iterrows()
            ]
            test_paths = [
                os.path.join(
                    self.dataroot,
                    row["slide_path"],
                    "patch_512",
                    "HE",
                    row["file_path"],
                )
                for _, row in df_test.iterrows()
            ]
        elif self.load_size == 1024:
            df_train = df[
                (df["patch_type"] == "upperpatch_1024") & (df["split"] == "train")
            ]
            df_test = df[
                (df["patch_type"] == "upperpatch_1024") & (df["split"] == "test")
            ]
            train_paths = [
                os.path.join(
                    self.dataroot,
                    row["slide_path"],
                    "upperpatch_1024",
                    "HE",
                    row["file_path"],
                )
                for _, row in df_train.iterrows()
            ]
            test_paths = [
                os.path.join(
                    self.dataroot,
                    row["slide_path"],
                    "upperpatch_1024",
                    "HE",
                    row["file_path"],
                )
                for _, row in df_test.iterrows()
            ]
        elif self.load_size == 2048:
            train_paths = []
            for train_slide in os.listdir(os.path.join(self.dataroot, "train")):
                patch_dir = os.path.join(
                    self.dataroot, "train", train_slide, "region_2048", "HE"
                )
                if os.path.exists(patch_dir):
                    train_paths += [
                        os.path.join(patch_dir, file_name)
                        for file_name in os.listdir(patch_dir)
                    ]
            test_paths = []
            for test_slide in os.listdir(os.path.join(self.dataroot, "test")):
                patch_dir = os.path.join(
                    self.dataroot, "test", test_slide, "region_2048", "HE"
                )
                if os.path.exists(patch_dir):
                    test_paths += [
                        os.path.join(patch_dir, file_name)
                        for file_name in os.listdir(patch_dir)
                    ]
        else:
            raise ValueError("load_size must be 256, 512, 1024, 2048")
        self.train_paths = []
        self.test_paths = []
        for filepath in train_paths:
            ihc_path = filepath.replace("HE", self.stain)
            if not os.path.exists(ihc_path):
                continue
            else:
                self.train_paths.append(filepath)
        for filepath in test_paths:
            ihc_path = filepath.replace("HE", self.stain)
            if not os.path.exists(ihc_path):
                continue
            else:
                self.test_paths.append(filepath)
        if self.train:
            self.file_paths = self.train_paths
        else:
            self.file_paths = self.test_paths

    def __getitem__(self, index):
        realA_path = self.file_paths[index]
        realB_path = realA_path.replace("HE", self.stain)
        slide_index = realA_path.split(os.sep)[-4]
        realA_img = Image.open(realA_path).convert("RGB")
        realB_img = Image.open(realB_path).convert("RGB")
        if self.train:
            params = self.get_params()
        else:
            params = {
                "crop_pos_A": (0, 0),
                "crop_pos_B": (0, 0),
                "flip": False,
                "needs_padding": False,
            }
        transform_A = self.get_transform(params, target="A")
        transform_B = self.get_transform(params, target="B")
        realA_tensor = transform_A(realA_img)
        realB_tensor = transform_B(realB_img)
        data = {
            "A": realA_tensor,
            "B": realB_tensor,
            "A_paths": realA_path,
            "B_paths": realB_path,
        }
        return data

    def __len__(self):
        return len(self.file_paths)

    def get_params(self):
        w, h = self.load_size, self.load_size
        margin_w = np.maximum(0, w - self.crop_size)
        margin_h = np.maximum(0, h - self.crop_size)
        flip = random.random() > 0.5 if (self.train and not self.no_flip) else False
        dx, dy = 0, 0
        if self.misalign > 0:
            target_distance = self.misalign * 0.8
            sigma = target_distance / 4.0
            r = random.gauss(target_distance, sigma)
            r = np.clip(r, 0, self.misalign)
            theta = random.uniform(0, 2 * np.pi)
            dx = int(r * np.cos(theta))
            dy = int(r * np.sin(theta))
            if margin_w > 0:
                dx = np.clip(dx, -margin_w, margin_w)
            if margin_h > 0:
                dy = np.clip(dy, -margin_h, margin_h)
        valid_x_min = max(0, -dx)
        valid_x_max = min(margin_w, margin_w - dx)
        valid_y_min = max(0, -dy)
        valid_y_max = min(margin_h, margin_h - dy)
        if valid_x_min <= valid_x_max:
            x_A = random.randint(valid_x_min, valid_x_max)
        else:
            x_A = 0
        if valid_y_min <= valid_y_max:
            y_A = random.randint(valid_y_min, valid_y_max)
        else:
            y_A = 0
        x_B = x_A + dx
        y_B = y_A + dy
        return {
            "crop_pos_A": (x_A, y_A),
            "crop_pos_B": (x_B, y_B),
            "flip": flip,
            "needs_padding": (self.load_size == self.crop_size) and (self.misalign > 0),
        }

    def get_transform(self, params, target="A", color_jitter=False):
        transform_list = []
        x, y = params[f"crop_pos_{target}"]
        crop_pos = (x, y)
        if params.get("needs_padding", False):
            padding = abs(self.misalign)
            transform_list.append(transforms.Pad(padding, padding_mode="reflect"))
            transform_list.append(
                transforms.Lambda(
                    lambda img: self.__crop(
                        img, (x + padding, y + padding), self.crop_size
                    )
                )
            )
        else:
            transform_list.append(
                transforms.Lambda(
                    lambda img: self.__crop(img, crop_pos, self.crop_size)
                )
            )
        if params["flip"]:
            transform_list.append(
                transforms.Lambda(lambda img: self.__flip(img, params["flip"]))
            )
        if color_jitter:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05
                )
            )
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)

    def __crop(self, img, pos, size):
        ow, oh = img.size
        x1, y1 = pos
        tw = th = size
        if ow > tw or oh > th:
            return img.crop((x1, y1, x1 + tw, y1 + th))
        return img

    def __flip(self, img, flip):
        if flip:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img
