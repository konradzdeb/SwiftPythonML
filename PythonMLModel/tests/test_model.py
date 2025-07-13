"""Test model on a few sample images."""

import os
import pytest
import numpy as np
from PIL import Image, ImageOps
import coremltools as ct
from tabulate import tabulate

@pytest.fixture(scope="module")
def model():
    model_path = os.path.join(os.path.dirname(__file__), "../FashionMNISTClassifier.mlpackage")
    return ct.models.MLModel(model_path)

def preprocess_image(image_path):
    with Image.open(image_path) as img:
        if img.mode != "L":
            img = img.convert("L")
        img = ImageOps.invert(img)
        img = img.resize((28, 28))
        arr = np.asarray(img).astype(np.float32) / 255.0
        return arr.flatten().tolist()

results = []

# Create parametrized test for different image files
@pytest.mark.parametrize("filename", ["t-shirt.jpeg", "pullover.jpg", "sneaker.jpg", "shirt.jpg"])
def test_model_prediction(filename, model):
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    img_path = os.path.join(fixtures_dir, filename)
    arr = preprocess_image(img_path)
    input_data = {f"pixel_{i}": val for i, val in enumerate(arr)}
    expected_label = os.path.splitext(filename)[0]
    output = model.predict(input_data)
    predicted_label = str(output["classLabel"])
    match = predicted_label.lower() == expected_label.lower()

    results.append((filename, expected_label, predicted_label, "✅" if match else "❌"))
    assert match, f"{filename}: expected {expected_label}, got {predicted_label}"

def pytest_sessionfinish(session, exitstatus):
    if results:
        print("\n\nModel Prediction Results:\n")
        print(tabulate(results, headers=["Filename", "Expected", "Predicted", "Match"]))
