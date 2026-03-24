import cv2
import numpy as np
from traditional.preprocessing import roi, clahe, gaussian_blur
from traditional.edge_detection import canny_by_otsu
from traditional.morphology import morphology_process
from traditional.segmentation import *
from traditional.visualization import draw_text


# def run_main_pipeline(image_path):
#     img = cv2.imread(image_path, 0)

#     roi_img = roi(img)
    
#     enhanced = clahe(roi_img)
#     blurred = gaussian_blur(enhanced)
#     edges = canny_by_otsu(blurred)

#     closed = morphology_process(edges)

#     mask = contour_mask(closed)
#     final = refine_mask(mask)

#     dist = distance_transform(final)
#     region = region_mask(dist)
#     segmented = segment_bone(roi_img, region)

#     # normalize dist
#     dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
#     dist_norm = dist_norm.astype("uint8")

#     # Ve grid
#     img1 = draw_text(roi_img, "1. ROI")
#     img2 = draw_text(enhanced, "2. CLAHE")
#     img3 = draw_text(dist_norm, "3. Distance")
#     seg_color = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)
#     img4 = draw_text(segmented, "4. Segmentation")

#     top = np.hstack((img1, img2))
#     bottom = np.hstack((img3, img4))
#     combined = np.vstack((top, bottom))

#     return combined, final



def run_main_pipeline(image_path):
    # Đọc ảnh xám
    img = cv2.imread(image_path, 0)
    if img is None: return None

    # Tiền xử lý
    roi_img = roi(img)
    enhanced = clahe(roi_img)
    blurred = gaussian_blur(enhanced)
    
    edges = canny_by_otsu(blurred)

    # Morphology với Adaptive Kernel
    closed = morphology_process(edges)

    # Tạo mặt nạ thô
    mask = contour_mask(closed)
    final = refine_mask(mask)

    # Phân vùng nâng cao bằng Watershed
    dist = distance_transform(final)
    # Gọi hàm Watershed mới
    final_region, segmented = apply_watershed(dist, final, roi_img)

    # --- Chuẩn bị hiển thị ---
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    
    img1 = draw_text(roi_img, "1. ROI")
    img2 = draw_text(enhanced, "2. CLAHE")
    img3 = draw_text(dist_norm, "3. Distance Map")
    img4 = draw_text(segmented, "4. Watershed Seg")

    # Ghép lưới hiển thị
    top = np.hstack((img1, img2))
    bottom = np.hstack((img3, img4))
    combined = np.vstack((top, bottom))

    return combined, final_region