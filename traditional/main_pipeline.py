import cv2
import numpy as np
from traditional.preprocessing import roi, clahe, gaussian_blur
from traditional.edge_detection import canny
from traditional.morphology import morphology_process
from traditional.segmentation import *
from traditional.visualization import draw_text


def run_main_pipeline(image_path):
    img = cv2.imread(image_path, 0)

    roi_img = roi(img)
    enhanced = clahe(roi_img)
    blurred = gaussian_blur(enhanced)
    edges = canny(blurred)

    dilated, closed = morphology_process(edges)

    mask = contour_mask(dilated)
    final = refine_mask(mask)

    dist = distance_transform(final)
    region = region_mask(dist)
    segmented = segment_bone(roi_img, region)

    # normalize dist
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
    dist_norm = dist_norm.astype("uint8")

    # Ve grid
    img1 = draw_text(roi_img, "1. ROI")
    img2 = draw_text(enhanced, "2. CLAHE")
    img3 = draw_text(final, "3. Final Mask")
    seg_color = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)
    seg_color[region > 0] = [0, 0, 255]  # tô đỏ vùng xương
    img4 = draw_text(segmented, "4. Segmentation")

    top = np.hstack((img1, img2))
    bottom = np.hstack((img3, img4))
    combined = np.vstack((top, bottom))

    return combined, final