"""
predict.py

COMP3710 – Inference & Visualization for Improved 3D UNet
---------------------------------------------------------
Loads the trained model checkpoint (unet3d_best.pth),
performs segmentation on unseen data, computes Dice coefficient,
and visualizes the MRI + label + prediction slice.

Author: Taokai Xing
"""

import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from dataset import Prostate3DDataset
from modules import ImprovedUNet3D



# Dice Metric (for evaluation)

def dice_coefficient(pred, target, num_classes):
    pred = torch.argmax(pred, dim=1)
    dice_scores = []
    for c in range(1, num_classes):  # skip background (0)
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        inter = (pred_c * target_c).sum()
        denom = pred_c.sum() + target_c.sum()
        dice_scores.append((2 * inter) / (denom + 1e-5))
    return torch.mean(torch.tensor(dice_scores))



#  Predict & Visualize Function

def predict_and_visualize(
    image_dir=r"D:\37100\data\HipMRI_study_complete_release_v1\semantic_MRs_anon",
    label_dir=r"D:\37100\data\HipMRI_study_complete_release_v1\semantic_labels_anon",
    checkpoint_path="unet3d_best.pth",
    sample_index=0,
    save_output=True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")

    # Load dataset
    dataset = Prostate3DDataset(image_dir, label_dir, training=False)
    image, label = dataset[sample_index]
    image = image.unsqueeze(0).to(device)  # [1, 1, H, W, D]
    label = label.to(device).long().squeeze(0)

    # Load trained model safely
    model = ImprovedUNet3D(in_channels=1, out_channels=6, deep_supervision=True)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Predict
    with torch.no_grad():
        pred = model(image)
        dice = dice_coefficient(pred, label.unsqueeze(0), num_classes=6).item()
    print(f" Dice score on sample {sample_index}: {dice:.4f}")

    # Convert prediction to numpy
    pred_label = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()


    #  Visualization: MRI slice, Ground Truth, Prediction

    mid_slice = image.shape[4] // 2
    img_np = image.cpu().squeeze(0).squeeze(0).numpy()
    lbl_np = label.cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_np[:, :, mid_slice], cmap="gray")
    plt.title("MRI Slice")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(lbl_np[:, :, mid_slice], cmap="nipy_spectral")
    plt.title("Ground Truth Label")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_label[:, :, mid_slice], cmap="nipy_spectral")
    plt.title("Predicted Segmentation")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("prediction_visual.png")
    plt.show()


    #  Optionally save predicted mask as NIfTI

    if save_output:
        output_dir = "predictions"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"prediction_{sample_index}.nii.gz")
        nib.save(nib.Nifti1Image(pred_label.astype(np.uint8), np.eye(4)), output_path)
        print(f" Saved predicted mask to: {output_path}")




#  Run Example

if __name__ == "__main__":
    predict_and_visualize(sample_index=0)
