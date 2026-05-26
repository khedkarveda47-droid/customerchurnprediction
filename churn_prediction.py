import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("churn.csv")

# Remove customerID column
data = data.drop(columns=["customerID"])

# Convert TotalCharges to numeric
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

# Remove missing values
data = data.dropna()

# Convert all columns to string first
for col in data.columns:
    data[col] = data[col].astype(str)

# Label Encoding all columns
encoder = LabelEncoder()

for col in data.columns:
    data[col] = encoder.fit_transform(data[col])

# Features and target
X = data.drop(columns=["Churn"])

y = data["Churn"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = DecisionTreeClassifier()

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("\nModel Saved Successfully!")