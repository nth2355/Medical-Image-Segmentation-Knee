import cv2
import numpy as np

def contour_mask(image, area_thresh=500):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > area_thresh:
            x, y, w_box, h_box = cv2.boundingRect(c)
            ratio = h_box / w_box
            if ratio > 10 and w_box < 20:
                continue
                
            cv2.drawContours(mask, [c], -1, 255, -1)
    return mask


def refine_mask(mask):
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=3)
    final = cv2.erode(mask, kernel, iterations=3)

    return final


def distance_transform(mask):

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    return dist


def region_mask(dist):

    _, region = cv2.threshold(dist, 0.1 * dist.max(), 255, 0)
    return region.astype("uint8")


def segment_bone(image, region):
    return cv2.bitwise_and(image, image, mask=region)

def apply_watershed(dist, final_mask, roi_img):

    # 1. Sure foreground (lõi xương)
    _, sure_fg = cv2.threshold(dist, 0.1 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype("uint8")

    # 2. Sure background
    kernel = np.ones((3,3), np.uint8)
    sure_bg = cv2.dilate(final_mask, kernel, iterations=2)

    # 3. Unknown
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 4. Marker
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 5. Watershed
    roi_color = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(roi_color, markers)

    # 6. Mask cuối (CHỈ lấy vùng xương)
    res_mask = np.zeros(roi_img.shape, dtype="uint8")
    res_mask[markers > 1] = 255

    segmented = cv2.bitwise_and(roi_img, roi_img, mask=res_mask)

    return res_mask, segmented