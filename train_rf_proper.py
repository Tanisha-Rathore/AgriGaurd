import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("merge.csv")

# Drop missing
df = df.dropna()

# Encode categorical features
crop_encoder = LabelEncoder()
district_encoder = LabelEncoder()

df["Crop_Encoded"] = crop_encoder.fit_transform(df["Crop"])
df["District_Encoded"] = district_encoder.fit_transform(df["District"])

# Features and target
X = df[["Crop_Encoded", "District_Encoded", "Year", "Area"]]
y = df["Production"]

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# Save model and encoders
joblib.dump(model, "models/rf_model_proper.pkl")
joblib.dump(crop_encoder, "models/crop_encoder.pkl")
joblib.dump(district_encoder, "models/district_encoder.pkl")

print("✅ Proper RF model trained and saved.")


