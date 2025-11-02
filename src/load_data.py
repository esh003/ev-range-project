import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- Load and prepare data ---
df = pd.read_csv("../data/electric_vehicles_spec_2025.csv")

cols = [
    "battery_capacity_kWh",
    "top_speed_kmh",
    "efficiency_wh_per_km",
    "acceleration_0_100_s",
    "length_mm",
    "width_mm",
    "height_mm"
]
target = "range_km"

df = df[cols + [target]].dropna()

# --- Train model ---
X = df.drop(columns=["range_km"])
y = df["range_km"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
