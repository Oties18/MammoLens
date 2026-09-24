"""Dataset and dataloader construction.

Expects CSV files with at least two columns (configurable):
  - image_path: path to a mammogram image (absolute, or relative to data_dir)
  - label:      integer class id (0 = benign, 1 = malignant)

Using CSV manifests rather than ImageFolder keeps train/val/test splits
explicit and reproducible, which matters for a medical dataset.
"""
from __future__ import annotations

import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MammoDataset(Dataset):
    def __init__(self, csv_path, data_dir, image_size, path_column,
                 label_column, train=False):
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.path_column = path_column
        self.label_column = label_column
        self.transform = build_transforms(image_size, train=train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row[self.path_column]
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.data_dir, img_path)
        # Mammograms are grayscale; convert to 3-channel so we can reuse
        # ImageNet-pretrained EfficientNet weights without surgery.
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = int(row[self.label_column])
        return image, label


def build_transforms(image_size, train=False):
    """ImageNet normalization (pretrained weights expect these stats)."""
    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        norm,
    ])


def make_loader(cfg, split):
    """Build a DataLoader for 'train', 'val', or 'test'."""
    csv_path = getattr(cfg.paths, f"{split}_csv")
    is_train = split == "train"
    dataset = MammoDataset(
        csv_path=csv_path,
        data_dir=cfg.paths.data_dir,
        image_size=cfg.data.image_size,
        path_column=cfg.data.path_column,
        label_column=cfg.data.label_column,
        train=is_train,
    )
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=is_train,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=is_train,
    )
