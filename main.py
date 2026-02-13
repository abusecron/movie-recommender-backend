from fastapi import FastAPI
import os
import joblib
import numpy as np

app = FastAPI()

ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__), "model_artifacts")

print("Loading film_enc...")
film_enc = joblib.load(f"{ARTIFACTS_PATH}/film_enc.pkl")
print(f"film_enc loaded: {len(film_enc.classes_)} classes")

print("Loading film_data...")
film_data = joblib.load(f"{ARTIFACTS_PATH}/film_data.pkl")
print(f"film_data loaded: {len(film_data)} rows")

print("Loading global_mean...")
global_mean = joblib.load(f"{ARTIFACTS_PATH}/global_mean.pkl")
print("global_mean loaded")

print("Loading item_factors...")
item_factors = joblib.load(f"{ARTIFACTS_PATH}/item_factors.pkl")
print(f"item_factors loaded: {item_factors.shape}")

print("Loading content_matrix...")
content_matrix = joblib.load(f"{ARTIFACTS_PATH}/content_matrix.pkl")
print(f"content_matrix loaded: {content_matrix.shape}")

print("All loaded!")

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/api/health")
def health():
    return {"status": "ok", "n_films": len(film_data)}
