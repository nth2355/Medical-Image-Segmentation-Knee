import sys
import os
import uuid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, request
import cv2

# from traditional.pipeline import traditional_pipeline
from deeplearning.predict import predict_mask, overlay_mask
from evaluation.metrics import segmentation_metrics
from evaluation.evaluate_dataset import evaluate_dataset
from traditional.main_pipeline import run_main_pipeline

app = Flask(__name__)

# -----------------------------
# folders
# -----------------------------

UPLOAD_FOLDER = "ui/static/uploads"
RESULT_FOLDER = "ui/static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


# -----------------------------
# MAIN PAGE
# -----------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    result = None

    if request.method == "POST":

        file = request.files["image"]
        method = request.form["method"]

        # -----------------------------
        # generate unique filename
        # -----------------------------

        uid = str(uuid.uuid4())
        filename = f"{uid}_{file.filename}"

        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)

        img = cv2.imread(input_path)

        # -----------------------------
        # find ground truth mask
        # -----------------------------

        gt_path = os.path.join(
            "dataset/roboflow/test/masks",
            file.filename  # dùng tên gốc để match dataset
        )

        gt_mask = None

        if os.path.exists(gt_path):
            gt_mask = cv2.imread(gt_path, 0)
            gt_mask = (gt_mask > 127).astype("uint8")

        # =============================
        # U-NET
        # =============================

        if method == "unet":

            mask = predict_mask(img)

            # metrics
            metrics = None
            if gt_mask is not None:
                metrics = segmentation_metrics(mask, gt_mask)

            # save
            mask_path = os.path.join(RESULT_FOLDER, f"mask_{uid}.png")
            overlay_path = os.path.join(RESULT_FOLDER, f"overlay_{uid}.png")

            cv2.imwrite(mask_path, mask * 255)

            overlay = overlay_mask(img, mask)
            cv2.imwrite(overlay_path, overlay)

            result = {
                "input": f"static/uploads/{filename}",
                "mask": f"static/results/mask_{uid}.png",
                "overlay": f"static/results/overlay_{uid}.png",
                "metrics": metrics,
                "method": "unet"
            }

        # =============================
        # TRADITIONAL
        # =============================

        else:

            # steps = traditional_pipeline(img)
            # pipeline_paths = {}

            # for name, image in steps.items():

            #     save_path = os.path.join(RESULT_FOLDER, f"{name}_{uid}.png")

            #     if image.max() <= 1:
            #         image = image * 255

            #     cv2.imwrite(save_path, image)
            #     pipeline_paths[name] = f"static/results/{name}_{uid}.png"

            # # final mask
            # final_mask = list(steps.values())[-1]

            # if final_mask.max() > 1:
            #     final_mask = final_mask / 255

            # final_mask = (final_mask > 0.5).astype("uint8")

            # # metrics
            # metrics = None
            # if gt_mask is not None:

            #     # resize pred về size của gt
            #     final_mask = cv2.resize(
            #         final_mask,
            #         (gt_mask.shape[1], gt_mask.shape[0]),
            #         interpolation=cv2.INTER_NEAREST
            #     )

            #     metrics = segmentation_metrics(final_mask, gt_mask)

            # result = {
            #     "input": f"static/uploads/{filename}",
            #     "pipeline": pipeline_paths,
            #     "metrics": metrics,
            #     "method": "traditional"
            # }
            combined, final_mask = run_main_pipeline(input_path)

            uid = str(uuid.uuid4())

            grid_path = os.path.join(RESULT_FOLDER, f"grid_{uid}.png")

            cv2.imwrite(grid_path, combined)
            
            # metrics
            metrics = None
            if gt_mask is not None:

                # resize pred về size của gt
                final_mask = cv2.resize(
                    final_mask,
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

                metrics = segmentation_metrics(final_mask, gt_mask)

            result = {
                    "input": f"static/uploads/{filename}",
                    "grid": f"static/results/grid_{uid}.png",
                    "method": "traditional",
                    "metrics": metrics 
             }

    return render_template("index.html", result=result)


# -----------------------------
# RUN SERVER
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True)