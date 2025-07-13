"""Convert a trained scikit-learn model to CoreML format for Swift integration."""

import coremltools as ct
from coremltools import colorlayout
import joblib
from datetime import datetime

# Load the trained model
model = joblib.load("fashion_mnist_rf_model.joblib")

# Define input and output features
# Assuming the model expects a 1x28x28 grayscale image as input
# and outputs a class label as a string.
input_features = [("image", ct.models.datatypes.Array(1, 28, 28))]
output_features = [("classLabel", ct.models.datatypes.String())]

# Convert to Core ML
coreml_model = ct.converters.sklearn.convert(model, input_features, output_features)
coreml_model.author = 'The Final Artefact'
coreml_model.version = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
coreml_model.short_description = 'Funny Fashion MNIST classifier'
coreml_model.output_description['classLabel'] = 'Predicted fashion item class'

# Save Core ML model
coreml_model.save("FashionMNISTClassifier.mlpackage")
print("CoreML model saved to: FashionMNISTClassifier.mlpackage")