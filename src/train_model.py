import pandas as pd

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
print("Shape:", df.shape)
print(df.head())

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(random_state=42)

# Define the hyperparameter grid
param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

# Randomized search for faster tuning
search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=10,
    scoring='r2',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

print("\nBest parameters:", search.best_params_)

best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

print("Tuned MAE:", mean_absolute_error(y_test, y_pred))
print("Tuned R²:", r2_score(y_test, y_pred))

import joblib
import os

# ensure models folder exists
os.makedirs("../models", exist_ok=True)

# save tuned model
joblib.dump(best_model, "../models/ev_range_model.joblib")
print("\n✅ Model saved successfully in /models/ev_range_model.joblib")

import matplotlib.pyplot as plt
import numpy as np

# --- Feature importance ---
feat_names = X.columns.tolist()
importances = best_model.feature_importances_
order = np.argsort(importances)[::-1]

plt.figure()
plt.bar(range(len(importances)), importances[order])
plt.xticks(range(len(importances)), [feat_names[i] for i in order], rotation=45, ha='right')
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.show()

# --- Predicted vs Actual ---
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.7)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims)  # ideal line
plt.xlabel("Actual range_km")
plt.ylabel("Predicted range_km")
plt.title("Predicted vs Actual")
plt.tight_layout()
plt.show()




