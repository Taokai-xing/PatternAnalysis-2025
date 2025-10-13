"""
train.py

COMP3710 – Prostate Segmentation using Improved 3D UNet
--------------------------------------------------------
Trains the model on HipMRI prostate dataset using Dice + CrossEntropy loss.
Includes GPU support, checkpoint saving, and validation Dice score.

Author: Taokai Xing
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

from dataset import Prostate3DDataset
from modules import ImprovedUNet3D


# ---------------------------------------------------------
# ⚙️ Dice + CrossEntropy Loss
# ---------------------------------------------------------
class DiceCELoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.smooth = smooth

    def forward(self, preds, targets):
        ce_loss = self.ce(preds, targets)
        preds_soft = torch.softmax(preds, dim=1)
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes=preds.shape[1])
        targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).float()

        intersection = (preds_soft * targets_onehot).sum(dim=(2, 3, 4))
        union = preds_soft.sum(dim=(2, 3, 4)) + targets_onehot.sum(dim=(2, 3, 4))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()

        return ce_loss + dice_loss


# ---------------------------------------------------------
# ⚙️ Compute Dice Coefficient for Evaluation
# ---------------------------------------------------------
def dice_coefficient(pred, target, num_classes):
    pred = torch.argmax(pred, dim=1)
    dice_scores = []
    for c in range(1, num_classes):  # skip background 0
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        inter = (pred_c * target_c).sum()
        denom = pred_c.sum() + target_c.sum()
        dice_scores.append((2 * inter) / (denom + 1e-5))
    return torch.mean(torch.tensor(dice_scores))


# ---------------------------------------------------------
# 🧠 Main Training Loop
# ---------------------------------------------------------
def train_model(
    image_dir=r"D:\37100\data\HipMRI_study_complete_release_v1\semantic_MRs_anon",
    label_dir=r"D:\37100\data\HipMRI_study_complete_release_v1\semantic_labels_anon",
    epochs=10,
    batch_size=1,
    lr=1e-4,
    val_split=0.1,
    save_path="unet3d_best.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Dataset
    full_dataset = Prostate3DDataset(image_dir, label_dir, training=True)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=1)

    # Model + Optimizer
    model = ImprovedUNet3D(in_channels=1, out_channels=6, deep_supervision=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = DiceCELoss()

    best_dice = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        with tqdm(train_dl, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}/{epochs}")
            for imgs, lbls in tepoch:
                imgs, lbls = imgs.to(device), lbls.long().to(device).squeeze(1)

                preds = model(imgs)
                loss = criterion(preds, lbls)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for imgs, lbls in val_dl:
                imgs, lbls = imgs.to(device), lbls.long().to(device).squeeze(1)
                preds = model(imgs)
                val_dice += dice_coefficient(preds, lbls, num_classes=6).item()

        val_dice /= len(val_dl)
        print(f"Epoch {epoch}: Train Loss={train_loss/len(train_dl):.4f}, Val Dice={val_dice:.4f}")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model (Dice={best_dice:.4f})")

    print(f"🎯 Training Done. Best Validation Dice: {best_dice:.4f}")


# ---------------------------------------------------------
# 🧪 Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    train_model(epochs=10, batch_size=1)

