import json
import pickle
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from torch import nn

# ==================== CONSTANTES Y MODELO ====================

DATA_DIR = Path(".")
EMB_DIM = 32  # debe coincidir con el embedding que generaste en deals.csv


class RecNet(nn.Module):
    """
    Modelo de recomendación que usa:
    - user_id (embedding)
    - deal_id (embedding)
    - role (embedding: buyer/seller)
    - user_cont: address_lat, address_lng
    - deal_cont: price promo, regular_price, lat, lng, distancia, descuento
    - image_embedding: vector de imagen del deal
    """
    def __init__(self, n_users, n_deals, n_roles, emb_dim_img=EMB_DIM):
        super().__init__()

        self.user_emb = nn.Embedding(n_users, 32)
        self.deal_emb = nn.Embedding(n_deals, 32)
        self.role_emb = nn.Embedding(n_roles, 8)

        # user_cont (2) + deal_cont (6) + user_emb(32) + deal_emb(32)
        # + role_emb(8) + img_emb(EMB_DIM)
        input_size = 2 + 6 + 32 + 32 + 8 + emb_dim_img

        self.mlp = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # logit de probabilidad de interés
        )

    def forward(self, batch):
        u_idx = batch["user_idx"]
        d_idx = batch["deal_idx"]
        role_idx = batch["role_idx"]
        user_cont = batch["user_cont"]
        deal_cont = batch["deal_cont"]
        img_emb = batch["img_emb"]

        u_e = self.user_emb(u_idx)
        d_e = self.deal_emb(d_idx)
        r_e = self.role_emb(role_idx)

        x = torch.cat([user_cont, deal_cont, u_e, d_e, r_e, img_emb], dim=1)
        out = self.mlp(x).squeeze(1)
        return out


# ==================== CARGA DE MAPEOS, CSVs Y MODELO ====================

# mappings.pkl
with open(DATA_DIR / "mappings.pkl", "rb") as f:
    mappings = pickle.load(f)

user_id_to_idx = mappings["user_id_to_idx"]
deal_id_to_idx = mappings["deal_id_to_idx"]
role_to_idx = mappings["role_to_idx"]

# deals.csv
deals_df = pd.read_csv(DATA_DIR / "deals.csv")


def parse_image_embedding(emb_str: str) -> np.ndarray:
    try:
        emb = json.loads(emb_str)
    except json.JSONDecodeError:
        emb = [0.0] * EMB_DIM
    if len(emb) != EMB_DIM:
        emb = (emb + [0.0] * EMB_DIM)[:EMB_DIM]
    return np.array(emb, dtype="float32")


deals_df["image_embedding_vec"] = deals_df["image_embedding"].apply(parse_image_embedding)
deals_df = deals_df.set_index("id")  # acceso rápido por deal_id

# tamaños
num_users = len(user_id_to_idx)
num_deals = len(deal_id_to_idx)
num_roles = len(role_to_idx)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RecNet(
    n_users=num_users,
    n_deals=num_deals,
    n_roles=num_roles,
    emb_dim_img=EMB_DIM,
).to(device)

state_dict = torch.load(DATA_DIR / "recommender_model.pt", map_location=device)
model.load_state_dict(state_dict)
model.eval()


# ==================== FUNCIONES AUXILIARES ====================

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Distancia aproximada en km entre dos coordenadas."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def _get_role_idx(role: str) -> int:
    role_lower = role.lower()
    if role_lower not in role_to_idx:
        if "buyer" in role_to_idx:
            return role_to_idx["buyer"]
        return list(role_to_idx.values())[0]
    return role_to_idx[role_lower]


def build_batch(
    user_id: str,
    buyer_lat: float,
    buyer_lng: float,
    role: str,
    deal_id: str,
):
    """
    Construye el batch para UN par (user, deal), con la misma lógica
    que usaste al entrenar RecDataset.
    """
    if user_id not in user_id_to_idx:
        raise KeyError(f"user_id {user_id} no existe en mappings.")

    if deal_id not in deal_id_to_idx:
        raise KeyError(f"deal_id {deal_id} no existe en mappings.")

    if deal_id not in deals_df.index:
        raise KeyError(f"deal_id {deal_id} no existe en deals.csv.")

    # índices de embeddings
    u_idx = user_id_to_idx[user_id]
    d_idx = deal_id_to_idx[deal_id]
    role_idx = _get_role_idx(role)

    # datos del deal
    d_row = deals_df.loc[deal_id]
    promo_price = float(d_row["price"])
    regular_price = float(d_row.get("regular_price", promo_price))
    d_lat = float(d_row["location_lat"])
    d_lng = float(d_row["location_lng"])

    # misma distancia que en el entrenamiento (euclídea en grados)
    dist = ((buyer_lat - d_lat) ** 2 + (buyer_lng - d_lng) ** 2) ** 0.5
    if regular_price > 0:
        discount = 1.0 - (promo_price / regular_price)
    else:
        discount = 0.0

    img_emb = d_row["image_embedding_vec"].astype("float32")

    # user_cont: [lat, lng]
    user_cont = np.array([buyer_lat, buyer_lng], dtype="float32")

    # deal_cont: [promo_price, regular_price, d_lat, d_lng, dist, discount]
    deal_cont = np.array(
        [promo_price, regular_price, d_lat, d_lng, dist, discount],
        dtype="float32",
    )

    # tensores (batch de tamaño 1)
    batch = {
        "user_idx": torch.tensor([u_idx], dtype=torch.long, device=device),
        "deal_idx": torch.tensor([d_idx], dtype=torch.long, device=device),
        "role_idx": torch.tensor([role_idx], dtype=torch.long, device=device),
        "user_cont": torch.tensor([user_cont], dtype=torch.float32, device=device),
        "deal_cont": torch.tensor([deal_cont], dtype=torch.float32, device=device),
        "img_emb": torch.tensor([img_emb], dtype=torch.float32, device=device),
    }

    # info extra para respuesta
    distance_km = haversine_km(buyer_lat, buyer_lng, d_lat, d_lng)
    category = str(d_row["category"])
    image_url = str(d_row["image_url"])

    return batch, distance_km, promo_price, regular_price, category, image_url


# ==================== DEFINICIÓN DE LA API ====================

app = FastAPI(title="Recommender API", version="1.0.0")


# ---------- Endpoint original: un solo deal ----------

class RecommendRequest(BaseModel):
    user_id: str
    address_lat: float
    address_lng: float
    role: str
    deal_id: str
    price: float  # precio que envía el frontend (referencial)


class RecommendResponse(BaseModel):
    user_id: str
    deal_id: str
    score: float           # 0–1, mayor = más recomendado
    distance_km: float
    deal_price: float
    regular_price: float
    category: str
    image_url: str


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    """
    Calcula la probabilidad (score) de que este user esté interesado en este deal.
    """
    try:
        batch, distance_km, promo_price, regular_price, category, image_url = build_batch(
            user_id=req.user_id,
            buyer_lat=req.address_lat,
            buyer_lng=req.address_lng,
            role=req.role,
            deal_id=req.deal_id,
        )
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))

    with torch.no_grad():
        logits = model(batch)
        prob = torch.sigmoid(logits).cpu().item()  # 0–1

    return RecommendResponse(
        user_id=req.user_id,
        deal_id=req.deal_id,
        score=prob,
        distance_km=distance_km,
        deal_price=promo_price,
        regular_price=regular_price,
        category=category,
        image_url=image_url,
    )


# ---------- NUEVO endpoint: varios deals recomendados ----------

class RecommendUsersRequest(BaseModel):
    user_id: str
    address_lat: float
    address_lng: float
    role: str = "buyer"
    category: Optional[str] = None   # ej. "ropa", "zapatillas", etc.
    top_n: int = 5                   # cuántos deals devolver
    radius_km: float = 5.0           # radio máximo en km


class RecommendUsersItem(BaseModel):
    user_id: str
    deal_id: str
    score: float
    distance_km: float
    deal_price: float
    regular_price: float
    category: str
    image_url: str


@app.post("/recommend_users", response_model=List[RecommendUsersItem])
def recommend_users(req: RecommendUsersRequest):
    """
    Devuelve varios deals distintos recomendados para un usuario,
    filtrando por categoría y radio (km) y ordenados por score.
    """
    # 1) Validaciones
    if req.user_id not in user_id_to_idx:
        raise HTTPException(status_code=400, detail="user_id no existe en mappings")

    role_idx = _get_role_idx(req.role)

    user_lat = float(req.address_lat)
    user_lng = float(req.address_lng)
    user_idx = user_id_to_idx[req.user_id]

    # 2) Candidatos: todos los deals, opcionalmente filtrando por categoría
    candidates = deals_df.copy()

    if req.category:
        candidates = candidates[candidates["category"] == req.category]

    if candidates.empty:
        raise HTTPException(status_code=404, detail="No hay deals para esa categoría")

    # 3) Distancia aproximada (grados -> km)
    dlat = candidates["location_lat"].astype(float) - user_lat
    dlng = candidates["location_lng"].astype(float) - user_lng
    dist_deg = (dlat**2 + dlng**2) ** 0.5
    dist_km = dist_deg * 111.0  # aprox

    candidates = candidates.assign(distance_km=dist_km)

    # Filtrar por radio si se indica
    if req.radius_km > 0:
        candidates = candidates[candidates["distance_km"] <= req.radius_km]

    if candidates.empty:
        raise HTTPException(status_code=404, detail="No hay deals dentro del radio indicado")

    # Para no reventar la máquina: nos quedamos con los 200 más cercanos
    candidates = candidates.sort_values("distance_km").head(200)

    # 4) Construir batch grande para el modelo
    user_idx_list = []
    deal_idx_list = []
    role_idx_list = []
    user_cont_list = []
    deal_cont_list = []
    img_emb_list = []
    distance_km_list = []

    for deal_id, row in candidates.iterrows():
        if deal_id not in deal_id_to_idx:
            continue

        promo_price = float(row["price"])
        regular_price = float(row.get("regular_price", promo_price))
        d_lat = float(row["location_lat"])
        d_lng = float(row["location_lng"])

        # misma distancia que en el entrenamiento (en grados)
        dist_deg_val = sqrt((user_lat - d_lat) ** 2 + (user_lng - d_lng) ** 2)
        dist_km_val = dist_deg_val * 111.0

        if regular_price > 0:
            discount = 1.0 - (promo_price / regular_price)
        else:
            discount = 0.0

        img_emb = row["image_embedding_vec"].astype("float32")

        user_idx_list.append(user_idx)
        deal_idx_list.append(deal_id_to_idx[deal_id])
        role_idx_list.append(role_idx)
        user_cont_list.append([user_lat, user_lng])
        deal_cont_list.append([
            promo_price,
            regular_price,
            d_lat,
            d_lng,
            dist_deg_val,
            discount,
        ])
        img_emb_list.append(torch.tensor(img_emb, dtype=torch.float32))
        distance_km_list.append(dist_km_val)

    if not user_idx_list:
        raise HTTPException(status_code=404, detail="No hay deals válidos para recomendar")

    batch = {
        "user_idx": torch.tensor(user_idx_list, dtype=torch.long, device=device),
        "deal_idx": torch.tensor(deal_idx_list, dtype=torch.long, device=device),
        "role_idx": torch.tensor(role_idx_list, dtype=torch.long, device=device),
        "user_cont": torch.tensor(user_cont_list, dtype=torch.float32, device=device),
        "deal_cont": torch.tensor(deal_cont_list, dtype=torch.float32, device=device),
        "img_emb": torch.stack(img_emb_list).to(device),
    }

    # 5) Pasar por el modelo
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits).cpu().numpy().tolist()

    # 6) Armar respuesta
    results: List[RecommendUsersItem] = []
    for (deal_id, row), score, dist_km_val in zip(candidates.iterrows(), probs, distance_km_list):
        results.append(
            RecommendUsersItem(
                user_id=req.user_id,
                deal_id=deal_id,
                score=float(score),
                distance_km=float(dist_km_val),
                deal_price=float(row["price"]),
                regular_price=float(row.get("regular_price", row["price"])),
                category=str(row["category"]),
                image_url=str(row["image_url"]),
            )
        )

    # ordenar por score desc y cortar top_n
    results.sort(key=lambda x: x.score, reverse=True)
    return results[: req.top_n]
