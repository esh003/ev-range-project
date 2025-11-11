import sys
from pathlib import Path
import streamlit as st

# --- import predict() from src/predict.py ---
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.append(str(SRC))
from predict import predict_range, FEATURES  # noqa

st.set_page_config(page_title="EV Range Predictor", page_icon="🚗", layout="centered")


st.title("🚗⚡ EV Range Predictor")

st.write("Enter specs and hit **Predict**.")

# Build inputs for each feature
vals = {}
vals["battery_capacity_kWh"]   = st.number_input("Battery capacity (kWh)",  min_value=10.0,  max_value=150.0, value=50.0,  step=0.1)
vals["top_speed_kmh"]          = st.number_input("Top speed (km/h)",         min_value=80.0,  max_value=300.0, value=180.0, step=1.0)
vals["efficiency_wh_per_km"]   = st.number_input("Efficiency (Wh/km)",       min_value=100.0, max_value=300.0, value=160.0, step=1.0)
vals["acceleration_0_100_s"]   = st.number_input("0–100 km/h (s)",           min_value=2.5,   max_value=20.0,  value=8.5,   step=0.1)
vals["length_mm"]              = st.number_input("Length (mm)",              min_value=3000.0,max_value=5500.0,value=4300.0,step=10.0)
vals["width_mm"]               = st.number_input("Width (mm)",               min_value=1500.0,max_value=2200.0,value=1800.0,step=5.0)
vals["height_mm"]              = st.number_input("Height (mm)",              min_value=1400.0,max_value=2000.0,value=1600.0,step=5.0)

if st.button("Predict"):
    try:
        pred_km = predict_range(vals)
        st.success(f"**Estimated range:** ~{pred_km:.1f} km")
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.subheader("ℹ️ About this project")
st.markdown("""
This project predicts the estimated driving range of an Electric Vehicle (EV) using Machine Learning.  
It uses:
- **XGBoost Regressor** (fine-tuned)
- Real EV dataset from **Kaggle**
- Features like battery capacity, efficiency, dimensions, speed, etc.

Built as part of the **AICTE – Green Technology + GenAI Internship**  
by Esha Bakshi 🔋⚡
""")
import re

st.markdown("---")
st.subheader("💬 EV Assistant (beta)")

q = st.text_input("Ask me something about EV range, battery, or your specs:")

FAQ = {
    "battery": "Battery capacity (kWh) is the energy stored. Higher kWh → generally higher range, but weight and efficiency still matter.",
    "efficiency": "Efficiency (Wh/km) is energy used per km. Lower Wh/km = better efficiency = more range from the same battery.",
    "acceleration": "Quicker 0–100 km/h (lower seconds) often means stronger motors; spirited driving reduces real-world range.",
    "lowrange": "If predicted range looks low, check efficiency (Wh/km) and height/drag. Try lower Wh/km, or bigger battery.",
    "tips": "Range tips: keep speeds moderate, gentle acceleration, proper tire pressure, reduce weight/roof racks, precondition battery in extreme temps.",
}

def quick_answer(q: str) -> str:
    s = q.lower()
    parts = []
    if "battery" in s:     parts.append(FAQ["battery"])
    if "efficiency" in s or "wh/km" in s: parts.append(FAQ["efficiency"])
    if "acceleration" in s or "0-100" in s: parts.append(FAQ["acceleration"])
    if "low" in s and "range" in s: parts.append(FAQ["lowrange"])
    if "tip" in s or "improve" in s: parts.append(FAQ["tips"])
    if not parts:
        return "Ask about battery, efficiency, acceleration, or give rough specs (e.g., '50 kWh, 160 Wh/km, 4300x1800x1600, 8.5s, 180 km/h')."
    return "\n\n".join(parts)


def try_predict_from_text(q: str):
    s = q.lower()

    def get_num(pat: str):
        m = re.search(pat, s)
        return float(m.group(1)) if m else None

    vals = {}
    vals["battery_capacity_kWh"] = get_num(r'(\d+(?:\.\d+)?)\s*kwh')
    vals["efficiency_wh_per_km"] = get_num(r'(\d+(?:\.\d+)?)\s*wh/?km')
    vals["top_speed_kmh"]        = get_num(r'(\d+(?:\.\d+)?)\s*km/?h\b')
    vals["acceleration_0_100_s"] = get_num(r'(\d+(?:\.\d+)?)\s*s(?![a-z])')

    # dimensions like 4300x1800x1600
    dims = re.search(r'(\d{3,5})\s*[x×]\s*(\d{3,5})\s*[x×]\s*(\d{3,5})', s)
    if dims:
        L, W, H = map(float, dims.groups())
        vals["length_mm"], vals["width_mm"], vals["height_mm"] = L, W, H

    # ensure we have all features
    if all(k in vals and vals[k] is not None for k in FEATURES):
        pred = predict_range(vals)
        return f"Estimated range ≈ **{pred:.1f} km** based on your text."
    return None


if q:
    pred_msg = try_predict_from_text(q)
    if pred_msg:
        st.success(pred_msg)
    else:
        st.info(quick_answer(q))

