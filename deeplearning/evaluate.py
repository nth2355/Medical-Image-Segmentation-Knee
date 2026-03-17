import torch
from torch.utils.data import DataLoader
import numpy as np

from dataset import KneeDataset
from model_unet import UNet

# cấu hình
MODEL_PATH = "saved_models/unet_knee.pth"
BATCH_SIZE = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# load test dataset
test_dataset = KneeDataset("dataset/roboflow/test")
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# hàm tính IoU
def compute_iou(pred, mask):
    intersection = (pred & mask).sum()
    union = (pred | mask).sum()
    return intersection / (union + 1e-6)

# hàm tính Dice
def compute_dice(pred, mask):
    intersection = (pred & mask).sum()
    return (2 * intersection) / (pred.sum() + mask.sum() + 1e-6)

total_iou = 0
total_dice = 0

total_tp = 0
total_fp = 0
total_fn = 0
total_tn = 0

num_samples = 0

with torch.no_grad():
    for batch_idx, (images, masks) in enumerate(test_loader):

        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)
        preds = (preds > 0.5).float()

        preds = preds.cpu().numpy()
        masks = masks.cpu().numpy()

        for i in range(len(preds)):
            pred = preds[i][0].astype(np.uint8)
            mask = masks[i][0].astype(np.uint8)

            # IoU & Dice
            intersection = np.logical_and(pred, mask).sum()
            union = np.logical_or(pred, mask).sum()

            iou = intersection / (union + 1e-6)
            dice = (2 * intersection) / (pred.sum() + mask.sum() + 1e-6)

            total_iou += iou
            total_dice += dice

            # confusion matrix pixel-wise
            tp = np.logical_and(pred == 1, mask == 1).sum()
            fp = np.logical_and(pred == 1, mask == 0).sum()
            fn = np.logical_and(pred == 0, mask == 1).sum()
            tn = np.logical_and(pred == 0, mask == 0).sum()

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            num_samples += 1

# ===== tính trung bình =====
mean_iou = total_iou / num_samples
mean_dice = total_dice / num_samples

precision = total_tp / (total_tp + total_fp + 1e-6)
recall = total_tp / (total_tp + total_fn + 1e-6)
accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + 1e-6)

print("\n===== MODEL EVALUATION =====")
print(f"Mean IoU     : {mean_iou:.4f}")
print(f"Mean Dice    : {mean_dice:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"Accuracy     : {accuracy:.4f}")