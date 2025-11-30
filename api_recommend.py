# api_recommend.py
import json
import pickle
from math import radians, sin, cos, asin, sqrt
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from torch import nn
from fastapi import FastAPI
from pydantic import BaseModel

# ===================== CONSTANTES =====================
DATA_DIR = Path(".")
EMB_DIM = 32
MODEL_PATH = DATA_DIR / "recommender_model.pt"
MAPPINGS_PATH = DATA_DIR / "mappings.pkl"
DEALS_PATH = DATA_DIR / "deals.csv"
USERS_PATH = DATA_DIR / "users.csv"

app = FastAPI(title="Recommender API")


# ===================== MODELO PYTORCH =================
class RecNet(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_deals: int,
        emb_user_dim: int = 32,
        emb_deal_dim: int = 32,
        emb_image_dim: int = EMB_DIM,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_user_dim)
        self.deal_emb = nn.Embedding(num_deals, emb_deal_dim)

        input_dim = emb_user_dim + emb_deal_dim + emb_image_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, user_idx, deal_idx, image_vec):
        u = self.user_emb(user_idx)
        d = self.deal_emb(deal_idx)
        x = torch.cat([u, d, image_vec], dim=-1)
        score = self.mlp(x).squeeze(-1)
        return score


# ===================== SCHEMAS FASTAPI =================
class RecommendRequest(BaseModel):
    user_id: str
    category: Optional[str] = None
    top_n: int = 5
    radius_km: float = 5.0


class RecommendItem(BaseModel):
    user_id: str
    deal_id: str
    score: float
    distance_km: float
    deal_price: float
    regular_price: float
    category: str


# ===================== UTILIDADES ======================
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Distancia entre dos puntos en km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


# ===================== CARGA GLOBAL ====================
print("Cargando mappings y modelo...")
with open(MAPPINGS_PATH, "rb") as f:
    mappings = pickle.load(f)

USER2IDX = mappings["user2idx"]
DEAL2IDX = mappings["deal2idx"]

MODEL = RecNet(
    num_users=len(USER2IDX),
    num_deals=len(DEAL2IDX),
    emb_user_dim=32,
    emb_deal_dim=32,
    emb_image_dim=EMB_DIM,
    hidden_dim=64,
)
state_dict = torch.load(MODEL_PATH, map_location="cpu")
MODEL.load_state_dict(state_dict)
MODEL.eval()

print("Cargando deals.csv...")
deals_df = pd.read_csv(DEALS_PATH)
deals_df["id"] = deals_df["id"].astype(str)

# Adaptado a TUS columnas de deals.csv
DEAL_LAT_COL = "location_lat"
DEAL_LNG_COL = "location_lng"
DEAL_PRICE_COL = "price"
DEAL_REG_PRICE_COL = "regular_price"
DEAL_CAT_COL = "category"

print("Cargando users.csv...")
users_df = pd.read_csv(USERS_PATH)
users_df["id"] = users_df["id"].astype(str)

# Adaptado a TUS columnas de users.csv
USER_LAT_COL = "address_lat"
USER_LNG_COL = "address_lng"

# Embeddings de imagen
DEAL_EMB_DICT = {}
for _, row in deals_df.iterrows():
    did = row["id"]
    emb_list = json.loads(row["image_embedding"])
    DEAL_EMB_DICT[did] = torch.tensor(emb_list, dtype=torch.float32)


# ===================== ENDPOINTS =======================
@app.get("/")
def root():
    return {"status": "ok", "message": "Recommender API running with REAL model"}


@app.post("/recommend_users", response_model=List[RecommendItem])
def recommend_users(req: RecommendRequest):
    # 1) Buscar al usuario
    user_row = users_df.loc[users_df["id"] == req.user_id]
    if user_row.empty:
        # Usuario no existe
        return []

    # Coordenadas del usuario
    user_lat = float(user_row[USER_LAT_COL].iloc[0])
    user_lng = float(user_row[USER_LNG_COL].iloc[0])

    # 2) Validar que el usuario existe en el modelo
    if req.user_id not in USER2IDX:
        # El modelo no vio este user_id en user_activity.csv
        return []

    u_idx = USER2IDX[req.user_id]

    # 3) Filtrar deals por categoría + radio
    candidates = []
    for _, row in deals_df.iterrows():
        deal_id = row["id"]

        if deal_id not in DEAL2IDX:
            continue

        # Filtro por categoría si se envió
        if req.category is not None:
            if str(row[DEAL_CAT_COL]).lower() != req.category.lower():
                continue

        deal_lat = float(row[DEAL_LAT_COL])
        deal_lng = float(row[DEAL_LNG_COL])

        distance = haversine_km(user_lat, user_lng, deal_lat, deal_lng)
        if distance > req.radius_km:
            continue

        candidates.append((deal_id, distance, row))

    if not candidates:
        return []

    deal_ids = [c[0] for c in candidates]
    distances = [c[1] for c in candidates]
    rows = [c[2] for c in candidates]

    # 4) Pasar por el modelo
    user_idx_tensor = torch.tensor([u_idx] * len(deal_ids), dtype=torch.long)
    deal_idx_tensor = torch.tensor([DEAL2IDX[d] for d in deal_ids], dtype=torch.long)
    img_vecs_tensor = torch.stack([DEAL_EMB_DICT[d] for d in deal_ids]).float()

    with torch.no_grad():
        scores = MODEL(user_idx_tensor, deal_idx_tensor, img_vecs_tensor)

    ranked = sorted(
        zip(deal_ids, distances, rows, scores.tolist()),
        key=lambda x: x[3],
        reverse=True,
    )

    top_k = min(req.top_n, len(ranked))
    result: List[RecommendItem] = []

    for deal_id, distance, row, score in ranked[:top_k]:
        deal_price = float(row[DEAL_PRICE_COL])
        regular_price = float(row[DEAL_REG_PRICE_COL])
        category = str(row[DEAL_CAT_COL])

        result.append(
            RecommendItem(
                user_id=req.user_id,
                deal_id=deal_id,
                score=float(score),
                distance_km=float(distance),
                deal_price=deal_price,
                regular_price=regular_price,
                category=category,
            )
        )

    return result
