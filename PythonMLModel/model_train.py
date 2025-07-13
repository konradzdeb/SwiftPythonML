"""Train a classifier on Fashion-MNIST and print performance summary."""

import joblib

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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


def dataset_to_numpy(dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Convert dataset to numpy arrays.

    Args:
        dataset (Dataset): The dataset to convert.

    Returns:
        tuple: A tuple containing the features and labels as numpy arrays.

    """
    x = dataset.data.numpy().reshape(len(dataset), -1)
    y = np.array([FASHION_LABELS[i] for i in dataset.targets.numpy()])
    return x, y


X_train_full, y_train_full = dataset_to_numpy(train_set)
X_test, y_test = dataset_to_numpy(test_set)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# Train classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_val)

print(f"Model Accuracy: {accuracy_score(y_val, y_pred):.3f}")

# Print sample predictions
sample_indices = np.random.choice(len(X_val), size=10, replace=False)
sample_images = X_val[sample_indices]
sample_true = y_val[sample_indices]
sample_pred = model.predict(sample_images)

df = pd.DataFrame({
    "True Label": sample_true,
    "Predicted Label": sample_pred,
    "Match": sample_true == sample_pred
})

print("\nSample Predictions:")
print(df.to_markdown(index=False))

# Export model
joblib.dump(model, "fashion_mnist_rf_model.joblib")
print("\nModel exported to: fashion_mnist_rf_model.joblib")
