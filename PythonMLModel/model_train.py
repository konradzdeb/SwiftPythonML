"""Train and evaluate a classifier, return metrics as pandas DataFrame."""

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target, name="target")
target_names = wine.target_names

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Classification report as dict → DataFrame
report_dict = classification_report(
    y_test, y_pred, target_names=target_names, output_dict=True
)
report_df = pd.DataFrame(report_dict).transpose().round(3)

# Optionally add accuracy as a separate row
report_df.loc["accuracy"] = ["", "", "", accuracy, ""]

# Display the final DataFrame
print(f"\nAccuracy: {accuracy:.3f}")
print("\nClassification Report (as DataFrame):")
print(report_df)
