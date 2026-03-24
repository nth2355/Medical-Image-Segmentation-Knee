import cv2
import numpy as np


def morphology_process(edges):

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

    dilated = cv2.dilate(edges, kernel, iterations=1)

    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)


    return closed



