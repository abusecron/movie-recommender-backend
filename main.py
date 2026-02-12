from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load artifacts ──────────────────────────────────────────────
ARTIFACTS_PATH = os.path.join(os.path.dirname(__file__),"model_artifacts")

print("Loading artifacts...")
item_factors = joblib.load(f"{ARTIFACTS_PATH}/item_factors.pkl")
item_factors = np.nan_to_num(item_factors.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

content_matrix = joblib.load(f"{ARTIFACTS_PATH}/content_matrix.pkl")
content_matrix = np.asarray(content_matrix, dtype=np.float64)

film_enc = joblib.load(f"{ARTIFACTS_PATH}/film_enc.pkl")
film_data = joblib.load(f"{ARTIFACTS_PATH}/film_data.pkl")
global_mean = joblib.load(f"{ARTIFACTS_PATH}/global_mean.pkl")

film_id_to_pos = {fid: i for i, fid in enumerate(film_data["id"])}
initial_films = pd.read_json(f"{ARTIFACTS_PATH}/initial_films.json")

# Precompute content_matrix indexed by film_enc order
print("Building content lookup...")
content_by_enc = np.zeros((len(film_enc.classes_), content_matrix.shape[1]), dtype=np.float64)
for fid, pos in film_id_to_pos.items():
    try:
        enc_idx = film_enc.transform([fid])[0]
        content_by_enc[enc_idx] = content_matrix[pos]
    except ValueError:
        pass

content_norms = np.linalg.norm(content_by_enc, axis=1)
content_norms[content_norms == 0] = 1.0

print(f"Ready! {len(film_data)} films loaded.")


# ── Schemas ─────────────────────────────────────────────────────
class RatedFilm(BaseModel):
    film_id: str
    liked: bool


class RecommendRequest(BaseModel):
    ratings: list[RatedFilm]
    n: int = 10


# ── Helpers ─────────────────────────────────────────────────────
def get_film_details(film_id: str) -> dict:
    pos = film_id_to_pos.get(film_id)
    if pos is None:
        return None
    row = film_data.iloc[pos]
    return {
        "film_id": film_id,
        "name": str(row.get("name", "")),
        "year": int(row["year"]) if pd.notna(row.get("year")) else None,
        "director": str(row.get("director", "")),
        "genres": str(row.get("genres", "")),
        "poster": str(row.get("poster", "")),
        "synopsis": str(row.get("synopsis", ""))[:200] if pd.notna(row.get("synopsis")) else "",
        "rating": round(float(row["rating"]), 2) if pd.notna(row.get("rating")) else None,
    }


def recommend_for_new_user(liked_ids, disliked_ids, n=10):
    liked_idxs = []
    disliked_idxs = []

    for fid in liked_ids:
        try:
            idx = int(film_enc.transform([fid])[0])
            if idx < item_factors.shape[0]:
                liked_idxs.append(idx)
        except ValueError:
            pass

    for fid in disliked_ids:
        try:
            idx = int(film_enc.transform([fid])[0])
            if idx < item_factors.shape[0]:
                disliked_idxs.append(idx)
        except ValueError:
            pass

    # SVD scoring
    svd_scores = np.zeros(item_factors.shape[0], dtype=np.float64)
    if liked_idxs:
        pseudo_user = np.mean(item_factors[liked_idxs], axis=0)
        if disliked_idxs:
            pseudo_user -= 0.5 * np.mean(item_factors[disliked_idxs], axis=0)
        pseudo_user = np.nan_to_num(pseudo_user, nan=0.0, posinf=0.0, neginf=0.0)
        svd_scores = pseudo_user @ item_factors.T
        svd_scores = np.nan_to_num(svd_scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Content scoring
    content_scores = np.zeros(item_factors.shape[0], dtype=np.float64)
    if liked_idxs:
        user_profile = np.mean(content_by_enc[liked_idxs], axis=0)
        if disliked_idxs:
            user_profile -= 0.5 * np.mean(content_by_enc[disliked_idxs], axis=0)
        user_profile = np.nan_to_num(user_profile, nan=0.0, posinf=0.0, neginf=0.0)
        user_norm = np.linalg.norm(user_profile)
        if user_norm > 0:
            content_scores = (content_by_enc @ user_profile) / (content_norms * user_norm)
            content_scores = np.nan_to_num(content_scores, nan=0.0, posinf=0.0, neginf=0.0)

    # Blend
    final_scores = 0.6 * svd_scores + 0.4 * content_scores

    for idx in liked_idxs + disliked_idxs:
        final_scores[idx] = -np.inf

    top_idxs = np.argsort(final_scores)[::-1][:n]
    top_film_ids = film_enc.inverse_transform(top_idxs)

    results = []
    for fid, score in zip(top_film_ids, final_scores[top_idxs]):
        details = get_film_details(fid)
        if details:
            details["score"] = round(float(score), 4)
            results.append(details)
    return results


# ── Routes ──────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "n_films": len(film_data)}


@app.get("/api/initial-films")
def get_initial_films():
    films = []
    for _, row in initial_films.iterrows():
        fid = row.get("id") or row.get("film_id")
        detail = get_film_details(fid)
        if detail:
            films.append(detail)

    if not films:
        sample = film_data[film_data["rating"].notna()].nlargest(20, "rating")
        for _, row in sample.iterrows():
            detail = get_film_details(row["id"])
            if detail:
                films.append(detail)

    return {"films": films}


@app.post("/api/recommend")
def recommend(request: RecommendRequest):
    liked = [r.film_id for r in request.ratings if r.liked]
    disliked = [r.film_id for r in request.ratings if not r.liked]

    if not liked and not disliked:
        return {"recommendations": [], "error": "Please rate at least one film."}

    results = recommend_for_new_user(liked, disliked, n=request.n)
    return {"recommendations": results}
