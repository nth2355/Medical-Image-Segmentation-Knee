import json
import cv2
import numpy as np
import os

def convert(dataset_path):

    annotation_file = os.path.join(dataset_path, "_annotations.coco.json")

    with open(annotation_file) as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]

    mask_dir = os.path.join(dataset_path, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    for img in images:

        img_id = img["id"]
        file_name = img["file_name"]
        height = img["height"]
        width = img["width"]

        mask = np.zeros((height, width), dtype=np.uint8)

        for ann in annotations:

            if ann["image_id"] == img_id:

                for seg in ann["segmentation"]:

                    poly = np.array(seg).reshape(-1,2).astype(np.int32)

                    cv2.fillPoly(mask, [poly], 255)

        cv2.imwrite(os.path.join(mask_dir, file_name), mask)

    print(f"Done: {dataset_path}")


if __name__ == "__main__":

    convert("dataset/roboflow/train")
    convert("dataset/roboflow/valid")
    convert("dataset/roboflow/test")