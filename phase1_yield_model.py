import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

print("🚀 Script started...")

# Load dataset
df = pd.read_csv("crop_production_new.csv")
print(df.columns.tolist()) 
# TEMP PROXY: use District as City (demo purpose)
df["City"] = df["District_Name"]

# Clean (Season added)
df = df.dropna(subset=["State_Name", "District_Name", "City", "Season", "Crop", "Crop_Year", "Area", "Production"])

# Features & target (Season + City added)
# X = df[["State_Name", "District_Name", "City", "Season", "Crop", "Crop_Year", "Area"]]
X = df[["District_Name", "Crop", "Crop_Year", "Area", "Rainfall_mm", "Temperature_C", "Season"]]
y = df["Production"]

# Categorical & numerical columns
# cat_cols = ["State_Name", "District_Name", "City", "Season", "Crop"]
# num_cols = ["Crop_Year", "Area"]
cat_cols = ["District_Name", "Crop", "Season"]
num_cols = ["Crop_Year", "Area", "Rainfall_mm", "Temperature_C"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

# (Optional) Speed-up for demo: sample 30% data
# df = df.sample(frac=0.3, random_state=42)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("⏳ Training city + season-level model...")
model = Pipeline(steps=[
    ("prep", preprocess),
    ("rf", RandomForestRegressor(n_estimators=15, random_state=42, n_jobs=-1))
])

# Train
model.fit(X_train, y_train)
print("✅ City + season-level model trained!")

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"📊 RMSE: {rmse:.2f}")
print(f"📈 R2: {r2:.3f}")

# Save
joblib.dump(model, "models/yield_rf_city_season_model.pkl")
print("💾 Model saved to models/yield_rf_city_season_model.pkl")
