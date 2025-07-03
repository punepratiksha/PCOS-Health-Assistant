# 🩺 PCOS Health Assistant – AI-Powered Health App

A smart, AI-powered Android application that predicts **Polycystic Ovary Syndrome (PCOS)** based on health inputs and symptoms. The app is backed by a machine learning model and provides personalized treatment recommendations to raise awareness and assist in early diagnosis.

## ❓ Problem & Solution

🩸 **Problem:** Millions of women suffer from PCOS without early diagnosis due to low awareness and limited access to healthcare, especially in rural areas.  
🌈 **Solution:** Swasthya Sakhi – a smart, accessible mobile assistant that uses AI to identify early signs of PCOS and guide users with helpful medical insights and lifestyle suggestions.

---

## 🚀 Features

- 🤖 **PCOS Prediction** using trained ML model
- 📱 **Android App** with user-friendly interface (Java + Retrofit)
- 🧠 **Backend in Flask** with model hosted on Render
- ✅ Inputs: Age, BMI, FSH, LH, Irregular Periods
- 🔍 Outputs: PCOS prediction + confidence + treatment tips
- 🔒 Firebase Authentication integration (optional)
- 🌐 API deployed and accessible in real-time

---

## 🧠 Machine Learning Model

- **Dataset Used:** `PCOS_extended_dataset.csv` (cleaned, balanced)
- **Algorithm:** RandomForestClassifier
- **Preprocessing:** Handled missing values, class imbalance (using SMOTE), and normalization
- **Accuracy Achieved:** `96.75%`
- **Model File:** `pcos_model.pkl`

### 🎯 Features Used for Prediction:
- Age  
- BMI  
- FSH (Follicle Stimulating Hormone)  
- LH (Luteinizing Hormone)  
- Irregular Periods (Yes/No)

---

## 📱 Android App Overview

- Built using **Android Studio**
- UI designed for easy input and instant feedback
- Communicates with Flask API using **Retrofit**
- Displays:
  - Prediction (Positive/Negative)
  - Confidence Score
  - Personalized Treatment Advice
- Future Plans: Add Firebase user history and health tracking
