
import cv2

def canny_by_otsu(image):
    t, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    lower = int(0.33 * t)
    upper = int(1.33 * t)
    edges = cv2.Canny(image, lower, upper)
    return edges

