"""
dataset.py

COMP3710 – 3D UNet Prostate Segmentation
Author: Taokai Xing
----------------------------------------
This file defines a PyTorch Dataset class to load 3D NIfTI medical images (.nii/.nii.gz)
from the HipMRI Study dataset. It supports normalization, random cropping,
and tensor conversion for training a 3D UNet.
"""

import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import random


class Prostate3DDataset(Dataset):
    def __init__(self, image_dir, label_dir, patch_size=(128, 128, 64), training=True):
        """
        Args:
            image_dir (str): path to folder with MRI volumes (.nii.gz)
            label_dir (str): path to folder with label volumes (.nii.gz)
            patch_size (tuple): optional crop size (HxWxD)
            training (bool): if True, apply random crop for training
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".nii") or f.endswith(".nii.gz")])
        self.patch_size = patch_size
        self.training = training

        assert len(self.image_files) == len(self.label_files), \
            f"Images ({len(self.image_files)}) and labels ({len(self.label_files)}) count mismatch!"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        lbl_path = os.path.join(self.label_dir, self.label_files[idx])

        # Load NIfTI image and label
        image = nib.load(img_path).get_fdata().astype(np.float32)
        label = nib.load(lbl_path).get_fdata().astype(np.uint8)

        # Normalize image (zero mean, unit variance)
        image = (image - np.mean(image)) / (np.std(image) + 1e-8)

        # Convert to torch tensor
        image = torch.from_numpy(image).unsqueeze(0)  # shape: [1, H, W, D]
        label = torch.from_numpy(label).unsqueeze(0)  # shape: [1, H, W, D]

        # Random crop during training
        if self.training and self.patch_size is not None:
            image, label = self.random_crop(image, label, self.patch_size)

        return image, label

    def random_crop(self, image, label, size):
        """Random 3D crop for data augmentation"""
        _, H, W, D = image.shape
        ph, pw, pd = size
        if H <= ph or W <= pw or D <= pd:
            return image, label  # skip if smaller than patch size

        # Choose random crop origin
        h0 = random.randint(0, H - ph)
        w0 = random.randint(0, W - pw)
        d0 = random.randint(0, D - pd)

        image = image[:, h0:h0 + ph, w0:w0 + pw, d0:d0 + pd]
        label = label[:, h0:h0 + ph, w0:w0 + pw, d0:d0 + pd]
        return image, label


# -------------------------------------------------------------------------
# 🔍 Quick test
# Run this file directly to verify loading works
# -------------------------------------------------------------------------
if __name__ == "__main__":
    image_dir = r"D:\37100\data\HipMRI_study_complete_release_v1\semantic_MRs_anon"
    label_dir = r"D:\37100\data\HipMRI_study_complete_release_v1\semantic_labels_anon"

    dataset = Prostate3DDataset(image_dir, label_dir, training=False)
    print(f"✅ Dataset loaded: {len(dataset)} samples found")

    # Get one sample
    img, lbl = dataset[0]
    print(f"Image shape: {img.shape}, Label shape: {lbl.shape}")
    print(f"Unique label values: {torch.unique(lbl)}")

    # Optional visualization (check one middle slice)
    import matplotlib.pyplot as plt
    mid_slice = img.shape[3] // 2
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img[0, :, :, mid_slice], cmap='gray')
    plt.title("MRI")
    plt.subplot(1, 2, 2)
    plt.imshow(lbl[0, :, :, mid_slice])
    plt.title("Label")
    plt.show()

