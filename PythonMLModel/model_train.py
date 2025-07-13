"""Train a classifier on Fashion-MNIST and print performance summary."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import coremltools as ct

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from torch.utils.data import Dataset
from torchvision import datasets, transforms


# Load Fashion-MNIST
transform = transforms.ToTensor()
train_set = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform,
)
test_set = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform,
)

# Use the class labels from the dataset
FASHION_LABELS = train_set.classes


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(test_set, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
print(classification_report(all_labels, all_preds, target_names=FASHION_LABELS))

example_input = torch.rand(1, 1, 28, 28).to(device)
traced = torch.jit.trace(model, example_input)
classifier_config = ct.ClassifierConfig(class_labels=FASHION_LABELS)
mlmodel = ct.convert(traced, inputs=[ct.ImageType(name="image", shape=(1, 1, 28, 28), scale=1/255.0)], classifier_config=classifier_config)
mlmodel.save("FashionMNISTClassifier.mlpackage")
print("Exported CoreML model to FashionMNISTClassifier.mlpackage")
