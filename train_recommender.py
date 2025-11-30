import json
import random
import pickle
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ===================== CONSTANTES =====================
DATA_DIR = Path(".")
EMB_DIM = 32          # debe coincidir con el embedding que generaste en deals.csv
BATCH_SIZE = 512
EPOCHS = 5
LR = 1e-3
RANDOM_SEED = 123

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ===================== MODELO =========================
class RecNet(nn.Module):
    """
    Modelo de recomendación que usa:
    - user_id (embedding)
    - deal_id (embedding)
    - image_embedding (vector de 32 floats)
    """

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
            nn.Linear(hidden_dim, 1),  # score
        )

    def forward(self, user_idx, deal_idx, image_vec):
        u = self.user_emb(user_idx)        # (B, emb_user_dim)
        d = self.deal_emb(deal_idx)        # (B, emb_deal_dim)
        x = torch.cat([u, d, image_vec], dim=-1)
        score = self.mlp(x).squeeze(-1)    # (B,)
        return score


# ===================== DATASET ========================
class InteractionsDataset(Dataset):
    def __init__(self, df_inter, deal_emb_dict, user2idx, deal2idx):
        """
        df_inter: dataframe con columnas user_id, deal_id
        deal_emb_dict: {deal_id: tensor(32)}
        user2idx / deal2idx: mapeos a índices
        """
        self.user_idx = []
        self.deal_idx = []
        self.image_vecs = []
        self.labels = []

        all_deal_ids = list(deal2idx.keys())

        # POSITIVOS
        for _, row in df_inter.iterrows():
            uid = row["user_id"]
            did = row["deal_id"]
            if uid not in user2idx or did not in deal2idx:
                continue
            u_idx = user2idx[uid]
            d_idx = deal2idx[did]
            img_vec = deal_emb_dict[did]

            self.user_idx.append(u_idx)
            self.deal_idx.append(d_idx)
            self.image_vecs.append(img_vec)
            self.labels.append(1.0)

            # NEGATIVO SIMPLE: mismo user, deal aleatorio distinto
            neg_did = random.choice(all_deal_ids)
            while neg_did == did:
                neg_did = random.choice(all_deal_ids)
            neg_d_idx = deal2idx[neg_did]
            neg_img_vec = deal_emb_dict[neg_did]

            self.user_idx.append(u_idx)
            self.deal_idx.append(neg_d_idx)
            self.image_vecs.append(neg_img_vec)
            self.labels.append(0.0)

        self.user_idx = torch.tensor(self.user_idx, dtype=torch.long)
        self.deal_idx = torch.tensor(self.deal_idx, dtype=torch.long)
        self.image_vecs = torch.stack(self.image_vecs).float()
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.user_idx[idx],
            self.deal_idx[idx],
            self.image_vecs[idx],
            self.labels[idx],
        )


# ===================== CARGA DE DATOS =================
def load_data():
    # ---- deals.csv ----
    deals_path = DATA_DIR / "deals.csv"
    deals = pd.read_csv(deals_path)

    # Nos aseguremos que id sea string
    deals["id"] = deals["id"].astype(str)

    # Parsear embedding de imagen
    deal_emb_dict = {}
    for _, row in deals.iterrows():
        did = row["id"]
        emb_list = json.loads(row["image_embedding"])
        # a tensor
        emb_tensor = torch.tensor(emb_list, dtype=torch.float32)
        deal_emb_dict[did] = emb_tensor

    # ---- user_activity.csv ----
    ua_path = DATA_DIR / "user_activity.csv"
    ua = pd.read_csv(ua_path)

    # Asegurar tipos string
    ua["user_id"] = ua["user_id"].astype(str)

    # Filtrar filas sin deal_id (NaN) o vacías
    ua = ua[ua["deal_id"].notna()]
    ua["deal_id"] = ua["deal_id"].astype(str)
    ua = ua[ua["deal_id"] != ""]

    # Solo nos quedamos con columnas relevantes
    ua = ua[["user_id", "deal_id"]]

    # ---- mapeos de IDs -> índices ----
    unique_user_ids = sorted(ua["user_id"].unique().tolist())
    unique_deal_ids = sorted(ua["deal_id"].unique().tolist())

    user2idx = {uid: idx for idx, uid in enumerate(unique_user_ids)}
    deal2idx = {did: idx for idx, did in enumerate(unique_deal_ids)}

    print(f"Usuarios únicos en interacciones: {len(user2idx)}")
    print(f"Deals únicos en interacciones: {len(deal2idx)}")

    # dataset
    dataset = InteractionsDataset(ua, deal_emb_dict, user2idx, deal2idx)
    return dataset, user2idx, deal2idx


# ===================== ENTRENAMIENTO ==================
def train():
    dataset, user2idx, deal2idx = load_data()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Entrenando en dispositivo:", device)

    model = RecNet(
        num_users=len(user2idx),
        num_deals=len(deal2idx),
        emb_user_dim=32,
        emb_deal_dim=32,
        emb_image_dim=EMB_DIM,
        hidden_dim=64,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for batch in dataloader:
            user_idx, deal_idx, img_vec, labels = batch
            user_idx = user_idx.to(device)
            deal_idx = deal_idx.to(device)
            img_vec = img_vec.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(user_idx, deal_idx, img_vec)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{EPOCHS} - Loss: {avg_loss:.4f}")

    # ===================== GUARDADO ===================
    # Guardamos mappings con ambas claves (nuevas y legacy)
    mappings = {
        "user2idx": user2idx,
        "deal2idx": deal2idx,
        "user_id_to_idx": user2idx,
        "deal_id_to_idx": deal2idx,
    }
    with open("mappings.pkl", "wb") as f:
        pickle.dump(mappings, f)

    # Guardamos SOLO los pesos (state_dict), no el modelo completo
    torch.save(model.state_dict(), "recommender_model.pt")
    print("Modelo y mappings guardados: recommender_model.pt, mappings.pkl")


if __name__ == "__main__":
    train()

