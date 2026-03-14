from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

print("🚀 Climate-aware model training started...")

# Load merged dataset
df = pd.read_csv("crop_with_rainfall.csv")
print("ALL COLUMNS:", df.columns.tolist())
print("ROWS BEFORE CLEANING:", len(df))

# Clean
# df = df.dropna(subset=[
#     "District_Name", "Crop", "Season",
#     "Crop_Year", "Area", "final_annual", "Production"
# ])
df = df.dropna(subset=[
    "District_Name", "Crop", "Season",
    "Crop_Year", "Area", "Production"
])
print("ROWS BEFORE CLEANING:", len(df))
print("NULL COUNTS:\n", df[[
    "District_Name", "Crop", "Season",
    "Crop_Year", "Area", "Production", "final_annual"
]].isna().sum())

# Features & Target
X = df[[
    "District_Name", "Crop", "Season",
    "Crop_Year", "Area", "Monsoon"
]]
y = df["Production"]

# Columns
# cat_cols = ["District_Name", "Crop", "Season"]
# num_cols = ["Crop_Year", "Area", "final_annual"]

# preprocess = ColumnTransformer(
#     transformers=[
#         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
#         ("num", "passthrough", num_cols)
#     ]
# )
# cat_cols = ["District_Name", "Crop", "Season"]
# num_cols = ["Crop_Year", "Area", "Monsoon"]



# preprocess = ColumnTransformer(
#     transformers=[
#         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
#         ("num", Pipeline(steps=[
#             ("imputer", SimpleImputer(strategy="median")),
#         ]), num_cols)
#     ]
# )
cat_cols = ["District_Name", "Crop", "Season"]
num_cols = ["Crop_Year", "Area"]

X = df[[
    "District_Name", "Crop", "Season",
    "Crop_Year", "Area"
]]
y = df["Production"]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Pipeline(steps=[
    ("prep", preprocess),
    ("rf", RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1))
])

print("⏳ Training model...")
model.fit(X_train, y_train)
print("✅ Model trained!")

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"📊 RMSE: {rmse:.2f}")
print(f"📈 R2: {r2:.3f}")

# Save model
joblib.dump(model, "models/yield_rf_climate_model.pkl")
print("💾 Climate-aware model saved to models/yield_rf_climate_model.pkl")
