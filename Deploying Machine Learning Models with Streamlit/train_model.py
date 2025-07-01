# train_model.py

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved as model.pkl")
