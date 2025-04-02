import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ✅ Load Dataset
file_path = r"C:\Users\prati\Downloads\archive (1)\PCOS_infertility.csv"
data = pd.read_csv(file_path)

# ✅ Preprocess Data
data = data.select_dtypes(include=['number'])
X = data.drop('PCOS (Y/N)', axis=1)  
y = data['PCOS (Y/N)']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# ✅ Train Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ✅ Evaluate Model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# ✅ Save Model
joblib.dump(clf, "pcos_model.pkl")
print("Model saved as pcos_model.pkl")
