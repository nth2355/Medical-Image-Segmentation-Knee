import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import KneeDataset
from model_unet import UNet

print("Starting training...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dataset
train_dataset = KneeDataset("dataset/roboflow/train")
val_dataset = KneeDataset("dataset/roboflow/valid")

print("Train images:", len(train_dataset))
print("Valid images:", len(val_dataset))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# model
model = UNet().to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 20

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS} starting...")

    model.train()
    train_loss = 0

    for batch_idx, (images, masks) in enumerate(train_loader):

        images = images.to(device)
        masks = masks.to(device)

        preds = model(images)

        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # in tiến trình từng batch
        if batch_idx % 5 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

    avg_loss = train_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} finished - Avg Loss: {avg_loss:.4f}")

# save model
torch.save(model.state_dict(), "saved_models/unet_knee.pth")

print("\nModel saved to saved_models/unet_knee.pth")