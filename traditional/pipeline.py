import cv2
import numpy as np


def traditional_pipeline(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    # 1 ROI
    roi = gray[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]

    # 2 CLAHE
    clahe = cv2.createCLAHE(clipLimit=9.0, tileGridSize=(18,18))
    enhanced = clahe.apply(roi)

    # 3 Gaussian
    blurred = cv2.GaussianBlur(enhanced,(5,5),0)

    # 4 Canny
    edges = cv2.Canny(blurred,40,120)

    # 5 Morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    dilated = cv2.dilate(edges,kernel,iterations=1)
    closed = cv2.morphologyEx(dilated,cv2.MORPH_CLOSE,kernel,iterations=2)

    # contour mask
    contours,_ = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(dilated)

    for c in contours:
        if cv2.contourArea(c) > 5000:
            cv2.drawContours(mask,[c],-1,255,-1)

    kernel = np.ones((7,7),np.uint8)

    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    final = cv2.dilate(mask,kernel,iterations=1)

    # 6 distance transform
    dist = cv2.distanceTransform(final,cv2.DIST_L2,5)

    dist_norm = cv2.normalize(dist,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)

    # 7 region mask
    _, region = cv2.threshold(dist,0.1*dist.max(),255,0)
    region = region.astype(np.uint8)

    # 8 segmentation
    segmented = cv2.bitwise_and(roi,roi,mask=region)

    return {
        "roi": roi,
        "clahe": enhanced,
        "blur": blurred,
        "edges": edges,
        "morph": final,
        "dist": dist_norm,
        "region": region,
        "segmented": segmented
    }