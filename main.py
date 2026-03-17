import cv2
import numpy as np

from traditional.preprocessing import roi, clahe, gaussian_blur
from traditional.edge_detection import canny
from traditional.morphology import morphology_process
from traditional.segmentation import *
from traditional.visualization import draw_text

SCALE_FACTOR = 0.1


img = cv2.imread("dataset/images/knee_xray3.jpg",0)

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


# ---------- HIỂN THỊ ----------

dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX)
dist_norm = dist_norm.astype("uint8")


img1 = draw_text(roi_img, "1. ROI")
img2 = draw_text(enhanced, "2. CLAHE")
img3 = draw_text(blurred, "3. Gaussian Blur")
img4 = draw_text(edges, "4. Canny Edge")
img5 = draw_text(closed, "5. Dilate")
img6 = draw_text(final, "6. Final Mask")
img7 = draw_text(dist_norm, "7. Distance Transform")
img8 = draw_text(region, "8. Region Mask")
img9 = draw_text(segmented, "9. Bone Segmentation")


top = np.hstack((img1,img2,img3))
mid = np.hstack((img4,img5,img6))
bot = np.hstack((img7,img8,img9))

combined = np.vstack((top,mid,bot))


# resize để không quá to
h,w = combined.shape[:2]

combined = cv2.resize(
    combined,
    (int(w*SCALE_FACTOR), int(h*SCALE_FACTOR)),
    interpolation=cv2.INTER_AREA
)


cv2.imshow("Knee Xray Segmentation", combined)

cv2.waitKey(0)

cv2.destroyAllWindows()