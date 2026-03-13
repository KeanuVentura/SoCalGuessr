"""My prediction code for the DSC 140B SoCalGuessr project.

This file is paired with `train.py`, which trains a model and saves its weights to
`mymodel.pt`. This file loads the saved model and uses it to make predictions on the test
set.

"""

import pathlib
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

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

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

def load_and_transform_image(path):

    image = Image.open(path).convert("RGB")
    pipeline = transforms.Compose([
        transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return pipeline(image).unsqueeze(0)

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights="DEFAULT")
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def predict(test_dir):

    test_dir = pathlib.Path(test_dir)

    model = ResNetClassifier(len(CLASSES))
    model.load_state_dict(torch.load("mymodel.pt"))
    model.eval()

    predictions = {}
    with torch.no_grad():
        for path in sorted(test_dir.glob("*.jpg")):
            image = load_and_transform_image(path)
            output = model(image)
            predicted_index = output.argmax(dim=1).item()
            predictions[path.name] = CLASSES[predicted_index]

    return predictions

if __name__ == "__main__":
    preds = predict("./testdata")
    print("Predictions:")
    for filename, label in sorted(preds.items()):
        print(f"{filename}: {label}")