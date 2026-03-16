import numpy as np

def segmentation_metrics(pred, gt):

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    TP = np.logical_and(pred, gt).sum()
    FP = np.logical_and(pred, np.logical_not(gt)).sum()
    FN = np.logical_and(np.logical_not(pred), gt).sum()
    TN = np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum()

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)
    dice = 2 * TP / (2 * TP + FP + FN + 1e-8)

    return {
        "accuracy": round(float(accuracy),4),
        "precision": round(float(precision),4),
        "recall": round(float(recall),4),
        "iou": round(float(iou),4),
        "dice": round(float(dice),4)
    }