# ⚡ EV Range Prediction using ML + Streamlit Chatbot

This project predicts the driving range (in km) of an electric vehicle based on its specifications.  
It combines a **machine learning model (XGBoost)** + a **Streamlit web app** + an optional **NLU chatbot mode** to help users understand EV specs in simple terms.

---

## 🚗 Features
- Predict EV range from:
  - Battery capacity (kWh)
  - Efficiency (Wh/km)
  - Top speed (km/h)
  - Acceleration (0–100 km/h)
  - Car dimensions (length, width, height)
- Live **Streamlit UI** for easy inputs
- **XGBoost model** with R² ~ **0.97**
- Built-in simple **chatbot** for EV-related questions
- Clean, modular code structure

---

## 🧠 Model Details
- **Algorithm:** XGBoost Regressor  
- **Dataset:** Electric vehicle specifications dataset (cleaned + feature-selected)  
- **Performance:**
  - MAE ≈ 12.1 km
  - R² ≈ 0.973

---

## 📂 Project Structure
├── data/
├── models/
│ └── ev_range_model.joblib
├── src/
│ ├── load_data.py
│ ├── train_model.py
│ └── predict.py
├── app.py
├── requirements.txt
└── README.md

---

## ▶️ Running the App

### 1. Install dependencies

### 2. Run the Streamlit app

---

## 🚀 Live Demo  
👉 **https://ev-range-project-id6ac4p2er39b8m6fyjc5m.streamlit.app**

---

## 💡 Example Prediction
Battery: 60 kWh
Efficiency: 150 Wh/km
Top speed: 180 km/h
Acceleration: 8.5 s
Dimensions: 4300 × 1800 × 1600 mm

→ Predicted range: ~310 km

---

## 🧑‍💻 Author
Esha Bakshi  
B.Tech, Manipal Institute of Technology  

---

