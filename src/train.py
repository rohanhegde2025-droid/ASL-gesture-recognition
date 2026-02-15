import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("asl-dataset/landmarks.csv")
print(f"Total samples loaded: {len(df)}")

X = df.drop("label", axis=1).values
y = df["label"].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")

print("\nTraining Random Forest...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

os.makedirs("models", exist_ok=True)
pickle.dump(clf, open("models/asl_classifier.pkl", "wb"))
pickle.dump(le, open("models/label_encoder.pkl", "wb"))

print("\nModel saved to models/asl_classifier.pkl")
print("Label encoder saved to models/label_encoder.pkl")