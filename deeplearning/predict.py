import torch
import cv2
import numpy as np
from torchvision import transforms
from deeplearning.model_unet import UNet

# cấu hình
MODEL_PATH = "saved_models/unet_knee.pth"
IMAGE_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load model

model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# preprocessing

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# predict mask

def predict_mask(image):

    # nếu ảnh là màu thì chuyển sang grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    original_h, original_w = image.shape[:2]

    img = transform(image)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img)

    pred = pred.squeeze().cpu().numpy()

    # threshold
    mask = (pred > 0.5).astype(np.uint8)*255

    # resize về kích thước ban đầu
    mask = cv2.resize(mask, (original_w, original_h))

    return mask

# overlay để hiển thị đẹp

def overlay_mask(image, mask):

    # nếu ảnh grayscale thì convert sang BGR để tô màu
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    colored_mask = np.zeros_like(image)

    # tô mask màu xanh
    colored_mask[mask > 0] = [0, 255, 0]

    overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)

    return overlay
# test nhanh

if __name__ == "__main__":

    img = cv2.imread(
        "dataset/roboflow/test/images/00000304_1x1_jpg.rf.e3f37b8f49fc76028e9ac1317c74c60f.jpg",
        cv2.IMREAD_GRAYSCALE
    )

    mask = predict_mask(img)
    overlay = overlay_mask(img, mask)

    cv2.imshow("Input", img)
    cv2.imshow("Mask", mask * 255)
    cv2.imshow("Overlay", overlay)

    cv2.waitKey(0)
    cv2.destroyAllWindows()