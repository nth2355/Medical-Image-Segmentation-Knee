import cv2

FONT_SCALE = 0.6
FONT_THICKNESS = 1


def draw_text(image, text):

    if len(image.shape) == 2:
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img = image.copy()

    cv2.putText(img, text, (15,25),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                (0,255,0),
                FONT_THICKNESS,
                cv2.LINE_AA)

    return img