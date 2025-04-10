import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# ✅ Load Trained Model
model_path = r"C:\Users\prati\Downloads\archive (1)\pcos_model.pkl"
model = joblib.load(model_path)
print("✅ Model loaded successfully!")

# ✅ Load Dataset for Feature Info
file_path = r"C:\Users\prati\Downloads\archive (1)\PCOS_infertility.csv"
data = pd.read_csv(file_path)
data = data.select_dtypes(include=['number'])

# ✅ Prepare New Data for Prediction (Using first row as example)
new_data = pd.DataFrame([data.iloc[0].drop("PCOS (Y/N)")])  

# ✅ Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.drop("PCOS (Y/N)", axis=1))

# ✅ Predict PCOS
prediction = model.predict(new_data)

print(f"🔍 **Prediction:** {'PCOS Detected' if prediction[0] == 1 else 'No PCOS Detected'}")
