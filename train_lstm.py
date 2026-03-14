import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# ===== 1. Load dataset =====
DATA_PATH = "crop_yield_timeseries.csv"  # 🔁 change if your file name is different
df = pd.read_csv(DATA_PATH)
print("Available Districts:", df["District"].unique()[:10])
print("Available Crops:", df["Crop"].unique()[:10])

# ===== 2. Filter one series (district + crop) =====
# ===== 2. Auto-pick a valid (District, Crop) with enough history =====
grouped = df.groupby(["District", "Crop"]).size().reset_index(name="count")

# Pick a pair with at least 6 data points (enough for LSTM window)
valid_pairs = grouped[grouped["count"] >= 6]

if valid_pairs.empty:
    raise ValueError("❌ No (District, Crop) pair has enough data for LSTM training.")

DISTRICT = valid_pairs.iloc[0]["District"]
CROP = valid_pairs.iloc[0]["Crop"]

print("✅ Using District:", DISTRICT, "| Crop:", CROP)

ts = df[(df["District"] == DISTRICT) & (df["Crop"] == CROP)].sort_values("Year")
yields = ts["Yield"].values.reshape(-1, 1)
# ===== 3. Scale =====
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(yields)

# ===== 4. Create sequences =====
def make_sequences(data, window=3):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

WINDOW = 3
X, y = make_sequences(y_scaled, WINDOW)

# ===== 5. Train-test split =====
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ===== 6. Build LSTM =====
model = Sequential([
    LSTM(64, input_shape=(WINDOW, 1)),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

# ===== 7. Train =====
model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

# ===== 8. Save model + scaler =====
os.makedirs("models", exist_ok=True)
model.save("models/lstm_yield_model.h5")
joblib.dump(scaler, "models/lstm_scaler.pkl")

print("✅ LSTM model trained & saved successfully")




