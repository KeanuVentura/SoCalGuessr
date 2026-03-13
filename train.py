"""My code for the DSC 140B SoCalGuessr project.

This code trains a ResNet18 model on the training images to classify which Southern 
California city a Street View image was taken in.

"""

import os
import pathlib
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

TRAIN_DIR = pathlib.Path("./data")

CLASSES = sorted(
    [
    "Anaheim",
    "Bakersfield",
    "Los_Angeles",
    "Riverside",
    "SLO",
    "San_Diego",
    ]
)

CLASS_TO_NUMBER = {name: i for i, name in enumerate(CLASSES)}

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 4

VALIDATION_FRACTION = 0.2

class SoCalDataset(Dataset):

    def __init__(self, root, transform=None):
        self.root = pathlib.Path(root)
        self.files = sorted(os.listdir(root))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        filename = self.files[idx]
        img_path = self.root / filename

        image = Image.open(img_path).convert("RGB")

        label_name = filename.split("-")[0]
        label = CLASS_TO_NUMBER[label_name]

        if self.transform:
            image = self.transform(image)

        return image, label

class ResNetClassifier(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.model = models.resnet18(weights="DEFAULT")

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def main():

    transform = transforms.Compose([
        transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    full_dataset = SoCalDataset(TRAIN_DIR, transform=transform)
    val_size = int(len(full_dataset) * VALIDATION_FRACTION)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_dataset.dataset.transform = transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = ResNetClassifier(len(CLASSES))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_acc = 0

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()

        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:

                outputs = model(images)

                preds = outputs.argmax(dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

        val_acc = val_correct / val_total
        val_accuracies.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "mymodel.pt")
            print("best model saved")

        print(
            f"Epoch {epoch+1}/{EPOCHS}  "
            f"Train Loss: {train_loss:.4f}  "
            f"Train Acc: {train_acc:.4f}  "
            f"Val Acc: {val_acc:.4f}"
        )

if __name__ == "__main__":
    main()
