import pandas as pd

crop = pd.read_csv("crop_production_new.csv")
rain = pd.read_csv("district_rainfall.csv")

# Rename columns to common names
crop = crop.rename(columns={
    "State_Name": "State",
    "District_Name": "District",
    "Crop_Year": "Year"
})

# Clean text
for df in [crop, rain]:
    df["State"] = df["State"].astype(str).str.strip().str.lower()
    df["District"] = df["District"].astype(str).str.strip().str.lower()
    df["Year"] = df["Year"].astype(int)

# Try merging again
merged = pd.merge(crop, rain, on=["State", "District", "Year"], how="inner")

print("Merged rows:", len(merged))
merged.to_csv("merge.csv", index=False)
