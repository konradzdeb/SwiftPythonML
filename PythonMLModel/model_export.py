

"""Convert a trained scikit-learn model to CoreML format for Swift integration."""

import coremltools as ct
import joblib

# Load the trained model
model = joblib.load("fashion_mnist_rf_model.joblib")

# Define example input shape (784 features for flattened 28x28 images)
input_features = [(f"pixel_{i}", ct.models.datatypes.Double()) for i in range(28 * 28)]
output_feature = "classLabel"

# Convert to Core ML
coreml_model = ct.converters.sklearn.convert(model, input_features, output_feature)

# Save Core ML model
coreml_model.save("FashionMNISTClassifier.mlpackage")
print("CoreML model saved to: FashionMNISTClassifier.mlpackage")