import cv2
import numpy as np


def contour_mask(image, area_thresh=5000):

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(image)

    for c in contours:

        if cv2.contourArea(c) > area_thresh:
            x, y, w_box, h_box = cv2.boundingRect(c)
            ratio = h_box / (w_box  +1e-5)
            if ratio > 5:
                continue
            cv2.drawContours(mask, [c], -1, 255, -1)

    return mask


def refine_mask(mask):

    kernel = np.ones((5,5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask = cv2.dilate(mask, kernel, iterations=1)

    k = np.ones((7,7), np.uint8)

    final = cv2.dilate(mask, k, iterations=2)

    return final


def distance_transform(mask):

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    return dist


def region_mask(dist):

    _, region = cv2.threshold(dist, 0.1 * dist.max(), 255, 0)

    return region.astype("uint8")


def segment_bone(image, region):

    return cv2.bitwise_and(image, image, mask=region)