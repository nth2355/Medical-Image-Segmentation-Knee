import os
import cv2

from deeplearning.predict import predict_mask
from evaluation.metrics import segmentation_metrics


def evaluate_dataset():

    img_dir = "dataset/roboflow/test/images"
    mask_dir = "dataset/roboflow/test/masks"

    iou_list = []
    dice_list = []

    for name in os.listdir(img_dir):

        img_path = os.path.join(img_dir, name)
        mask_path = os.path.join(mask_dir, name)

        if not os.path.exists(mask_path):
            continue

        img = cv2.imread(img_path)

        pred = predict_mask(img)

        gt = cv2.imread(mask_path, 0)
        gt = (gt > 127).astype("uint8")

        metrics = segmentation_metrics(pred, gt)

        iou_list.append(metrics["iou"])
        dice_list.append(metrics["dice"])

    mean_iou = sum(iou_list) / len(iou_list)
    mean_dice = sum(dice_list) / len(dice_list)

    return {
        "mean_iou": round(mean_iou, 4),
        "mean_dice": round(mean_dice, 4),
        "samples": len(iou_list)
    }