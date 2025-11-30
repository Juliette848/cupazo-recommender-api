import csv
import random
import json
import uuid
from datetime import datetime, timedelta

# ==============================
# CONFIG
# ==============================
N_ROWS = 50_000        # filas por tabla
RANDOM_SEED = 123
random.seed(RANDOM_SEED)

# Dimensión del embedding de imagen
EMB_DIM = 32   # puedes cambiar a 64, 128, etc.

# ==============================
# HELPERS
# ==============================
start_dt = datetime.now() - timedelta(days=365)
end_dt = datetime.now() + timedelta(days=30)
total_seconds = int((end_dt - start_dt).total_seconds())


def random_datetime_iso():
    """Fecha aleatoria entre start_dt y end_dt en ISO 8601."""
    return (start_dt + timedelta(seconds=random.randint(0, total_seconds))).isoformat()


def random_embedding(dim=EMB_DIM):
    """
    Genera un embedding aleatorio de dimensión 'dim'.
    Valores en el rango [-1, 1], redondeados a 6 decimales.
    """
    return [round(random.uniform(-1, 1), 6) for _ in range(dim)]


def make_uuid() -> str:
    """
    Genera un UUID4 válido para Postgres/Supabase.
    Ejemplo: 550e8400-e29b-41d4-a716-446655440000
    """
    return str(uuid.uuid4())


cities = ["Lima", "Arequipa", "Trujillo", "Cusco", "Piura"]
roles_user = ["buyer", "seller"]
status_group = ["open", "full", "completed", "cancelled"]
status_interest = ["interested", "maybe", "joined_group"]
time_windows = ["mañana", "tarde", "noche"]
event_types = ["view_deal", "click_deal", "join_group", "leave_group", "search", "view_list"]
sources = ["mobile_app", "web"]
device_types = ["mobile", "desktop"]
oses = ["Android", "iOS", "Windows", "macOS"]

# Para subcategorías de ropa en metadata (lo dejamos igual)
clothing_subcats = [
    "vestidos", "blusas", "pantalones", "faldas", "casacas",
    "cardigans", "tops", "leggings", "chaquetas", "conjuntos"
]

streets = [
    "Av. Primavera", "Av. Arequipa", "Av. Javier Prado",
    "Av. Brasil", "Av. Universitaria",
    "Jr. Los Olivos", "Jr. Miraflores",
    "Calle Las Flores", "Calle Los Pinos", "Calle Los Laureles"
]

# categorías para deals
deal_categories = [
    "ropa",
    "zapatillas",
    "accesorios",
    "decoracion",
    "joyas",
    "hecho_a_mano",
    "cuidado_personal",
    "bolsos",
]

colors = [
    "rojo", "azul", "amarillo", "negro", "blanco",
    "verde", "rosado", "marrón", "celeste"
]

# ==============================
# 1) users.csv
# ==============================
print("Generando users.csv ...")
user_ids = []
user_reliability = {}

with open("users.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "id", "name", "email", "address", "address_lat", "address_lng",
        "reliability_score", "role", "created_at", "city"
    ])

    for i in range(N_ROWS):
        uid = make_uuid()
        user_ids.append(uid)

        city = random.choice(cities)
        name = f"User_{i+1}"
        email = f"user{i+1}@example.com"
        address = f"{random.choice(streets)} {random.randint(100,999)}, {city}"

        # Coordenadas alrededor de Perú
        lat = -12.0464 + random.uniform(-5, 5)
        lng = -77.0428 + random.uniform(-5, 5)

        reliability = random.randint(1, 5)
        role = random.choice(roles_user)
        created_at = random_datetime_iso()

        writer.writerow([
            uid, name, email, address, lat, lng,
            reliability, role, created_at, city
        ])

        user_reliability[uid] = reliability

# ==============================
# 2) deals.csv
# ==============================
print("Generando deals.csv ...")
deal_ids = []
deal_max_group_size = {}
deal_category_map = {}

with open("deals.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "id", "user_id", "title", "description", "type",
        "max_group_size", "price", "category",
        "location_lat", "location_lng", "active",
        "created_at", "expires_at", "image_url",
        "image_embedding",   # embedding de imagen
        "regular_price"      # precio unitario normal
    ])

    for i in range(N_ROWS):
        did = make_uuid()
        deal_ids.append(did)

        seller_id = random.choice(user_ids)
        category = random.choice(deal_categories)
        color = random.choice(colors)

        # ==== TÍTULO Y DESCRIPCIÓN SEGÚN CATEGORÍA ====
        if category == "ropa":
            prenda = random.choice(
                ["vestido", "blusa", "pantalón", "falda", "casaca", "cardigan", "top", "legging", "chaqueta", "conjunto"]
            )
            size = random.choice(["XS", "S", "M", "L", "XL"])
            title = f"{prenda.capitalize()} color {color} talla {size}"
            description = (
                f"{prenda.capitalize()} de color {color} talla {size}, ideal para uso diario "
                f"y combinar con distintos outfits."
            )

        elif category == "zapatillas":
            shoe_size = random.randint(35, 42)
            title = f"Zapatillas color {color} talla {shoe_size}"
            description = (
                f"Zapatillas color {color}, talla {shoe_size}, cómodas y ligeras "
                f"para caminar o hacer deporte."
            )

        elif category == "accesorios":
            accesorio = random.choice(["collar", "pulsera", "aretes", "anillo"])
            title = f"{accesorio.capitalize()} color {color}"
            description = (
                f"{accesorio.capitalize()} color {color} con detalles delicados, "
                f"perfecto para complementar tu look."
            )

        elif category == "decoracion":
            deco = random.choice(["cuadro decorativo", "almohada decorativa", "lámpara de mesa"])
            title = f"{deco.capitalize()} color {color}"
            description = (
                f"{deco.capitalize()} en color {color}, ideal para darle estilo y personalidad "
                f"a tu hogar."
            )

        elif category == "joyas":
            joya = random.choice(["collar de plata", "anillo ajustable", "aretes con perlas"])
            title = f"{joya.capitalize()} color {color}"
            description = (
                f"{joya.capitalize()} en tono {color}, diseño elegante y moderno para ocasiones "
                f"especiales."
            )

        elif category == "hecho_a_mano":
            handmade = random.choice(["bolso tejido", "muñeco amigurumi", "portamacetas tejido"])
            title = f"{handmade.capitalize()} color {color}"
            description = (
                f"{handmade.capitalize()} hecho a mano en color {color}, pieza única "
                f"con acabado artesanal."
            )

        elif category == "cuidado_personal":
            cp = random.choice(["set de skincare", "crema hidratante", "kit de spa"])
            title = f"{cp.capitalize()} para uso diario"
            description = (
                f"{cp.capitalize()} pensado para el cuidado personal, ayuda a mantener "
                f"la piel suave y fresca."
            )

        elif category == "bolsos":
            bolso = random.choice(["bolso de mano", "mochila casual", "cartera pequeña"])
            title = f"{bolso.capitalize()} color {color}"
            description = (
                f"{bolso.capitalize()} de color {color}, práctico y espacioso para llevar "
                f"tus cosas esenciales."
            )

        else:
            # fallback por si acaso
            title = f"Producto de categoría {category} color {color}"
            description = f"Producto de categoría {category} en color {color}."

        # ==== RESTO DE CAMPOS DEL DEAL ====
        deal_type = random.choice(["2x1", "3x2", "group_price"])
        max_group = random.randint(2, 5)

        # regular_price = precio normal por 1 unidad
        regular_price = round(random.uniform(39.9, 299.9), 2)
        # price = precio PROMOCIONAL para compras grupales (más barato)
        discount_factor = random.uniform(0.6, 0.9)  # 10–40% aprox
        price = round(regular_price * discount_factor, 2)

        loc_lat = -12.0464 + random.uniform(-2, 2)
        loc_lng = -77.0428 + random.uniform(-2, 2)

        active = random.choice([True, True, True, False])
        created_at = random_datetime_iso()
        expires_at = (
            datetime.fromisoformat(created_at)
            + timedelta(days=random.randint(3, 60))
        ).isoformat()

        # ID de imagen "falsa"
        image_url = f"b64_img_{i+1:07d}"

        # Embedding "falso" para esa imagen
        image_emb = random_embedding()
        image_emb_str = json.dumps(image_emb, ensure_ascii=False)

        writer.writerow([
            did, seller_id, title, description, deal_type,
            max_group, price, category,
            loc_lat, loc_lng, active,
            created_at, expires_at, image_url,
            image_emb_str, regular_price
        ])

        deal_max_group_size[did] = max_group
        deal_category_map[did] = category

# ==============================
# 3) match_groups.csv
# ==============================
print("Generando match_groups.csv ...")
group_ids = []

with open("match_groups.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "deal_id", "max_group_size", "status", "created_at"])

    for i in range(N_ROWS):
        gid = make_uuid()
        group_ids.append(gid)

        deal_id = random.choice(deal_ids)
        max_group = deal_max_group_size[deal_id]
        status = random.choice(status_group)
        created_at = random_datetime_iso()

        writer.writerow([gid, deal_id, max_group, status, created_at])

# ==============================
# 4) match_group_members.csv
# ==============================
print("Generando match_group_members.csv ...")

with open("match_group_members.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "id", "group_id", "user_id", "role",
        "joined_at", "delivery_address", "delivery_lat", "delivery_lng", "status"
    ])

    for i in range(N_ROWS):
        mid = make_uuid()
        gid = random.choice(group_ids)
        uid = random.choice(user_ids)

        role = "owner" if random.random() < 0.2 else "buyer"
        joined_at = random_datetime_iso()

        city = random.choice(cities)
        delivery_address = f"Entrega en {random.choice(streets)} {random.randint(100,999)}, {city}"
        d_lat = -12.0464 + random.uniform(-2, 2)
        d_lng = -77.0428 + random.uniform(-2, 2)

        status = random.choice(["pending", "confirmed", "delivered", "cancelled"])

        writer.writerow([
            mid, gid, uid, role,
            joined_at, delivery_address, d_lat, d_lng, status
        ])

# ==============================
# 5) deal_interests.csv
# ==============================
print("Generando deal_interests.csv ...")

with open("deal_interests.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "id", "deal_id", "user_id", "status",
        "preferred_time_window", "created_at"
    ])

    for i in range(N_ROWS):
        iid = make_uuid()
        deal_id = random.choice(deal_ids)
        uid = random.choice(user_ids)

        status = random.choice(status_interest)
        tw = random.choice(time_windows)
        created_at = random_datetime_iso()

        writer.writerow([iid, deal_id, uid, status, tw, created_at])

# ==============================
# 6) user_activity.csv
# ==============================
print("Generando user_activity.csv ...")

with open("user_activity.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "id", "user_id", "deal_id",
        "event_type", "source", "metadata", "created_at"
    ])

    for i in range(N_ROWS):
        eid = make_uuid()
        uid = random.choice(user_ids)
        event_type = random.choice(event_types)
        source = random.choice(sources)
        created_at = random_datetime_iso()

        # Eventos ligados a deals
        if event_type in ["view_deal", "click_deal", "join_group", "leave_group"]:
            deal_id = random.choice(deal_ids)
            price = round(random.uniform(39.9, 299.9), 2)
            subcat = random.choice(clothing_subcats)
            max_group = deal_max_group_size[deal_id]
            position_in_list = random.randint(1, 20)
            deal_cat = deal_category_map[deal_id]

            metadata = {
                "session_id": f"sess_{random.randint(1, 1_000_000)}",
                "device": {
                    "type": random.choice(device_types),
                    "os": random.choice(oses),
                    "app_version": f"1.{random.randint(0,5)}.{random.randint(0,9)}"
                },
                "event_data": {
                    "deal_id": deal_id,
                    "deal_category": deal_cat,
                    "deal_subcategory": subcat,
                    "deal_price": price,
                    "deal_discount": round(random.uniform(0.1, 0.6), 2),
                    "list_type": random.choice(
                        ["home_recommended", "search_results", "similar_items"]
                    ),
                    "position_in_list": position_in_list,
                    "max_group_size": max_group,
                    "experiment_id": random.choice(
                        ["exp_home_v1", "exp_home_v2", "exp_search_v1"]
                    )
                }
            }
        else:
            # Eventos sin deal asociado (deal_id NULL en Supabase)
            deal_id = None
            metadata = {
                "session_id": f"sess_{random.randint(1, 1_000_000)}",
                "device": {
                    "type": random.choice(device_types),
                    "os": random.choice(oses),
                    "app_version": f"1.{random.randint(0,5)}.{random.randint(0,9)}"
                },
                "event_data": {
                    "query": random.choice(
                        ["vestidos", "blusas", "ofertas lima", "ropa deportiva", "casacas"]
                    ) if event_type == "search" else "",
                    "screen": random.choice(["home", "deal_list", "favorites", "profile"])
                }
            }

        metadata_str = json.dumps(metadata, ensure_ascii=False)
        writer.writerow([
            eid, uid, deal_id,
            event_type, source, metadata_str, created_at
        ])

# ==============================
# 7) match_recommendations.csv
# ==============================
print("Generando match_recommendations.csv ...")

with open("match_recommendations.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "id", "user_id", "candidate_id",
        "distance_km", "similarity", "reliability_score",
        "role", "rank", "created_at"
    ])

    for i in range(N_ROWS):
        rid = make_uuid()
        user_id = random.choice(user_ids)

        # aseguramos que candidate_id != user_id
        candidate_id = random.choice(user_ids)
        while candidate_id == user_id:
            candidate_id = random.choice(user_ids)

        distance_km = round(random.uniform(0.1, 30.0), 3)
        similarity = round(random.uniform(0.0, 1.0), 4)

        # tomamos el reliability del candidato como referencia
        reliability_score = user_reliability.get(candidate_id, random.randint(1, 5))

        role = random.choice(["buyer", "seller"])
        rank = random.randint(1, 10)
        created_at = random_datetime_iso()

        writer.writerow([
            rid, user_id, candidate_id,
            distance_km, similarity, reliability_score,
            role, rank, created_at
        ])

print("¡Listo! Se generaron los 7 CSV con 50,000 filas cada uno, usando UUIDs válidos para Supabase.")
