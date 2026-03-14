import pandas as pd

rain = pd.read_csv("district_rainfall.csv")
crop = pd.read_csv("crop_production_new.csv")

print("Rainfall columns:", rain.columns.tolist())
print("Crop columns:", crop.columns.tolist())

print("Rainfall sample:\n", rain.head())
print("Crop sample:\n", crop.head())
