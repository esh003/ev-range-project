from pathlib import Path
import joblib
import pandas as pd

# columns used during training (order matters)
FEATURES = [
    "battery_capacity_kWh",
    "top_speed_kmh",
    "efficiency_wh_per_km",
    "acceleration_0_100_s",
    "length_mm",
    "width_mm",
    "height_mm",
]

# load trained model
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "ev_range_model.joblib"
model = joblib.load(MODEL_PATH)

def predict_range(inputs: dict) -> float:
    """inputs: dict with the 7 feature keys (numbers). returns predicted range in km."""
    missing = [k for k in FEATURES if k not in inputs]
    if missing:
        raise ValueError(f"Missing keys: {missing}")
    row = {k: float(inputs[k]) for k in FEATURES}
    X = pd.DataFrame([row], columns=FEATURES)
    return float(model.predict(X)[0])

# quick manual test
if __name__ == "__main__":
    sample = {
        "battery_capacity_kWh": 50,
        "top_speed_kmh": 180,
        "efficiency_wh_per_km": 160,
        "acceleration_0_100_s": 8.5,
        "length_mm": 4300,
        "width_mm": 1800,
        "height_mm": 1600,
    }
    print("Predicted range (km):", round(predict_range(sample), 1))
