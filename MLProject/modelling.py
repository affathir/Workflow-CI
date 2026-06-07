import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import os

# Membaca data bersih dari folder yang sama
df = pd.read_csv('padi_preprocessing/padi_clean.csv')
X = df.drop('Produksi', axis=1)
y = df['Produksi']

# Training sederhana
model = RandomForestRegressor()
model.fit(X, y)

# Simpan model agar bisa digunakan di Kriteria 4
mlflow.sklearn.log_model(model, "model")
print("Training selesai dan model tersimpan!")