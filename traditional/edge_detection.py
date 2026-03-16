import cv2

def canny(image, low=60, high=120):
    edges = cv2.Canny(image, low, high)
    return edges