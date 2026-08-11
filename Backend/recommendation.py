import math
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

BACKEND_DIR = Path(__file__).resolve().parent
APP_DIR = BACKEND_DIR.parent
PROJECT_ROOT = APP_DIR.parent

DATASETS_DIR = PROJECT_ROOT / "Datasets"
SPOTS_CSV_CANDIDATES = [
    DATASETS_DIR / "other spots.csv",
    DATASETS_DIR / "other_spots.csv",
    APP_DIR / "Datasets" / "other spots.csv",
    BACKEND_DIR / "Datasets" / "other spots.csv",
    Path.cwd() / "Datasets" / "other spots.csv",
    Path.cwd().parent / "Datasets" / "other spots.csv"
]
SPOTS_CSV = next((p for p in SPOTS_CSV_CANDIDATES if p.exists()), DATASETS_DIR / "other spots.csv")

print("========== TOURIST DATASET ==========")
print("BACKEND_DIR:", BACKEND_DIR)
print("PROJECT_ROOT:", PROJECT_ROOT)
print("SPOTS_CSV:", SPOTS_CSV)
print("FILE EXISTS:", SPOTS_CSV.exists())

if not SPOTS_CSV.exists():
    raise FileNotFoundError(f"Tourist spots dataset not found: {SPOTS_CSV}")

spots_df = pd.read_csv(SPOTS_CSV)
spots_df.columns = spots_df.columns.astype(str).str.strip().str.lower()

print("TOTAL SPOTS:", len(spots_df))
print("COLUMNS:", spots_df.columns.tolist())

if "district" in spots_df.columns:
    print("DISTRICTS COUNT:", spots_df["district"].astype(str).str.strip().nunique())
    print("HYDERABAD SPOTS:", len(spots_df[spots_df["district"].astype(str).str.strip().str.casefold() == "hyderabad"]))
    print("BHADRADRI SPOTS:", len(spots_df[spots_df["district"].astype(str).str.strip().str.casefold() == "bhadradri kothagudem"]))
print("====================================")

# Secondary datasets loaded with candidates for nearby amenities
amenities_csv_candidates = [
    DATASETS_DIR / "nearby_amenities.csv",
    APP_DIR / "Datasets" / "nearby_amenities.csv",
    Path.cwd() / "Datasets" / "nearby_amenities.csv"
]
amenities_csv = next((p for p in amenities_csv_candidates if p.exists()), DATASETS_DIR / "nearby_amenities.csv")

acc_csv_candidates = [
    DATASETS_DIR / "accommodations.csv",
    APP_DIR / "Datasets" / "accommodations.csv",
    Path.cwd() / "Datasets" / "accommodations.csv"
]
acc_csv = next((p for p in acc_csv_candidates if p.exists()), DATASETS_DIR / "accommodations.csv")

visitors_csv_candidates = [
    DATASETS_DIR / "spot_visitors.csv",
    APP_DIR / "Datasets" / "spot_visitors.csv",
    Path.cwd() / "Datasets" / "spot_visitors.csv"
]
visitors_csv = next((p for p in visitors_csv_candidates if p.exists()), DATASETS_DIR / "spot_visitors.csv")

amenities_df = pd.read_csv(amenities_csv) if amenities_csv.exists() else pd.DataFrame()
acc_df = pd.read_csv(acc_csv) if acc_csv.exists() else pd.DataFrame()
visitors_df = pd.read_csv(visitors_csv) if visitors_csv.exists() else pd.DataFrame()

if not amenities_df.empty:
    amenities_df.columns = amenities_df.columns.astype(str).str.strip().str.lower()
if not acc_df.empty:
    acc_df.columns = acc_df.columns.astype(str).str.strip().str.lower()
if not visitors_df.empty:
    visitors_df.columns = visitors_df.columns.astype(str).str.strip().str.lower()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().casefold().split())


def get_all_districts() -> List[str]:
    """Returns a sorted list of unique districts from dataset."""
    if spots_df.empty or "district" not in spots_df.columns:
        raise RuntimeError("Tourist spots dataset is empty.")
    districts = spots_df["district"].dropna().astype(str).str.strip().unique().tolist()
    return sorted([d for d in districts if d])


def get_spots(district: Optional[str] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
    """Returns detailed spot items filtered by district and category."""
    if spots_df.empty:
        raise RuntimeError("Tourist spots dataset is empty.")

    df = spots_df.copy()

    if district and normalize_text(district) != "all":
        target = normalize_text(district)
        df = df[df["district"].apply(normalize_text) == target]

    if category and normalize_text(category) != "all":
        target_category = normalize_text(category)
        df = df[df["category"].apply(normalize_text) == target_category]

    spots = []
    for _, row in df.iterrows():
        lat = float(row.get("lat", 17.3850))
        lon = float(row.get("lon", 78.4867))

        if math.isnan(lat) or lat == 0:
            lat = 17.3850
        if math.isnan(lon) or lon == 0:
            lon = 78.4867

        reviews_val = row.get("reviews", 1500)
        try:
            reviews_cnt = int(float(reviews_val))
        except Exception:
            reviews_cnt = 1500

        spots.append({
            "name": str(row["name"]).strip(),
            "district": str(row["district"]).strip(),
            "category": str(row.get("category", "heritage")).strip(),
            "rating": float(row.get("rating", 4.5)),
            "popularity": float(row.get("popularity", 80.0)),
            "entry_fee": float(row.get("entry_fee", 0.0)),
            "lat": lat,
            "lon": lon,
            "reviews": reviews_cnt
        })

    spots.sort(key=lambda x: (x["rating"], x["popularity"]), reverse=True)
    print(f"[Spots] District={district} Returned={len(spots)}")
    return spots


def calculate_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine formula to compute distance in km between two coordinate points."""
    try:
        R = 6371.0  # Earth radius in km
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return round(R * c, 2)
    except Exception:
        return 2.5


def get_nearby_amenities(spot_name: str, district: str, lat: float, lon: float) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns categorized nearby amenities:
    hotels, restaurants, attractions, hospitals, parking, ATMs, petrol_pumps, restrooms.
    """
    results = {
        "hotels": [],
        "restaurants": [],
        "attractions": [],
        "hospitals": [],
        "parking": [],
        "atms": [],
        "petrol_pumps": [],
        "restrooms": []
    }

    # 1. Hotels from accommodations.csv
    if not acc_df.empty:
        dist_acc = (
            acc_df[acc_df["district"].astype(str).str.lower() == district.lower()]
            if "district" in acc_df.columns
            else acc_df
        )
        for _, r in dist_acc.iterrows():
            a_lat = float(r.get("lat", lat + 0.01))
            a_lon = float(r.get("lon", lon + 0.01))
            dist_km = calculate_distance_km(lat, lon, a_lat, a_lon)
            results["hotels"].append({
                "name": str(r.get("name", "Hotel")),
                "tier": str(r.get("tier", "Mid")),
                "cost": float(r.get("cost", 2500.0)),
                "rating": 4.4,
                "lat": a_lat,
                "lon": a_lon,
                "distance_km": dist_km,
                "address": f"Near {spot_name}, {district}"
            })

    # 2. Amenities from nearby_amenities.csv
    if not amenities_df.empty:
        matched_am = (
            amenities_df[
                (amenities_df["district"].astype(str).str.lower() == district.lower())
                | (amenities_df["spot_name"].astype(str).str.lower() == spot_name.lower())
            ]
            if "district" in amenities_df.columns
            else amenities_df
        )

        for _, r in matched_am.iterrows():
            am_type = str(r.get("amenity_type", "restaurant")).lower()
            a_lat = float(r.get("lat", lat + 0.005))
            a_lon = float(r.get("lon", lon + 0.005))
            dist_km = calculate_distance_km(lat, lon, a_lat, a_lon)

            item = {
                "name": str(r.get("amenity_name", f"{am_type.title()} Point")),
                "type": am_type,
                "rating": 4.3,
                "lat": a_lat,
                "lon": a_lon,
                "distance_km": dist_km,
                "address": f"{district} Central Road"
            }

            if "restaurant" in am_type or "food" in am_type:
                results["restaurants"].append(item)
            elif "hospital" in am_type or "clinic" in am_type:
                results["hospitals"].append(item)
            elif "atm" in am_type or "bank" in am_type:
                results["atms"].append(item)
            elif "parking" in am_type:
                results["parking"].append(item)
            elif "petrol" in am_type or "fuel" in am_type:
                results["petrol_pumps"].append(item)
            elif "restroom" in am_type or "toilet" in am_type:
                results["restrooms"].append(item)

    # 3. Nearby attractions from spots_df
    all_spots = get_spots(district=district)
    for s in all_spots:
        if s["name"].lower() != spot_name.lower():
            d_km = calculate_distance_km(lat, lon, s["lat"], s["lon"])
            results["attractions"].append({
                "name": s["name"],
                "category": s["category"],
                "rating": s["rating"],
                "entry_fee": s["entry_fee"],
                "lat": s["lat"],
                "lon": s["lon"],
                "distance_km": d_km,
                "address": f"{s['district']} Sightseeing"
            })

    # Synthetic fallback generator for complete UI map rendering
    def add_synth_amenities(key: str, count: int, name_prefix: str, amenity_type: str, lat_offset: float, lon_offset: float):
        if len(results[key]) < 2:
            for i in range(1, count + 1):
                m_lat = lat + (i * lat_offset)
                m_lon = lon + (i * lon_offset)
                results[key].append({
                    "name": f"{name_prefix} {spot_name} #{i}",
                    "type": amenity_type,
                    "rating": round(4.2 + (i * 0.1), 1),
                    "lat": m_lat,
                    "lon": m_lon,
                    "distance_km": round(0.5 * i, 1),
                    "address": f"Zone {i}, {district}"
                })

    add_synth_amenities("hotels", 3, "Grand Stay", "hotel", 0.008, 0.006)
    add_synth_amenities("restaurants", 3, "Spice Haven", "restaurant", -0.005, 0.007)
    add_synth_amenities("hospitals", 2, "City Care Hospital", "hospital", 0.012, -0.008)
    add_synth_amenities("parking", 2, "Tourist Parking Zone", "parking", -0.004, -0.004)
    add_synth_amenities("atms", 2, "State Bank ATM", "atm", 0.003, -0.005)
    add_synth_amenities("petrol_pumps", 2, "Indian Oil Station", "petrol_pump", -0.009, 0.003)
    add_synth_amenities("restrooms", 2, "Public Clean Restroom", "restroom", 0.002, 0.004)

    # Sort each list by distance
    for k in results:
        results[k] = sorted(results[k], key=lambda x: x.get("distance_km", 0.0))[:6]

    return results


def recommend_places(
    selected_spot: Optional[str] = None,
    district: Optional[str] = None,
    category: Optional[str] = None,
    season: Optional[str] = None,
    budget: float = 0.0,
    crowd: int = 0,
    transport: str = "car",
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Generates recommended tourist spots. If selected_spot is provided, calculates
    similarity based on category, district proximity (Haversine distance), and rating.
    """
    all_spots = get_spots(district=None, category=None)
    if not all_spots:
        return []

    # Find the target selected spot object if provided
    ref_spot = None
    if selected_spot:
        ref_spot = next((s for s in all_spots if s["name"].lower() == selected_spot.lower()), None)

    candidates = []

    if ref_spot:
        ref_lat = ref_spot["lat"]
        ref_lon = ref_spot["lon"]
        ref_cat = ref_spot["category"].lower()
        ref_dist = ref_spot["district"].lower()

        for s in all_spots:
            # Exclude the reference spot itself
            if s["name"].lower() == ref_spot["name"].lower():
                continue

            # Calculate Haversine distance from selected spot
            dist_km = calculate_distance_km(ref_lat, ref_lon, s["lat"], s["lon"])

            # Category similarity score (1.0 for exact match, 0.4 for partial)
            cat_score = 1.0 if s["category"].lower() == ref_cat else 0.4

            # District / Proximity score
            dist_score = 1.0 if s["district"].lower() == ref_dist else max(0.1, 1.0 / (1.0 + (dist_km / 40.0)))

            # Rating score
            rating_score = float(s["rating"]) / 5.0

            # Composite similarity score (0.0 to 1.0)
            similarity = (0.45 * cat_score) + (0.35 * dist_score) + (0.20 * rating_score)

            # Apply district/category filters if explicitly specified by user
            if district and district.lower() != "all" and s["district"].lower() != district.lower():
                continue
            if category and category.lower() != "all" and s["category"].lower() != category.lower():
                continue

            est_budget = budget if budget > 0 else float(s["entry_fee"] + 1200.0)
            exp_crowd = crowd if crowd > 0 else int(s["popularity"] * 150)

            candidates.append({
                "spot_name": s["name"],
                "district": s["district"],
                "category": s["category"].title(),
                "rating": s["rating"],
                "season": season or "All Seasons",
                "expected_crowd": exp_crowd,
                "estimated_budget": est_budget,
                "transport": transport,
                "lat": s["lat"],
                "lon": s["lon"],
                "entry_fee": s["entry_fee"],
                "reviews": s["reviews"],
                "distance_from_selected_km": dist_km,
                "similarity_match_percent": round(similarity * 100, 1),
                "reference_spot": ref_spot["name"]
            })

        # Sort candidates by similarity match percent descending
        candidates = sorted(candidates, key=lambda x: x["similarity_match_percent"], reverse=True)
        return candidates[:8]

    else:
        # Fallback ranking if no selected spot is passed
        filtered_spots = get_spots(district=district, category=category)
        for s in filtered_spots[:10]:
            est_budget = budget if budget > 0 else float(s["entry_fee"] + 1200.0)
            exp_crowd = crowd if crowd > 0 else int(s["popularity"] * 150)

            candidates.append({
                "spot_name": s["name"],
                "district": s["district"],
                "category": s["category"].title(),
                "rating": s["rating"],
                "season": season or "All Seasons",
                "expected_crowd": exp_crowd,
                "estimated_budget": est_budget,
                "transport": transport,
                "lat": s["lat"],
                "lon": s["lon"],
                "entry_fee": s["entry_fee"],
                "reviews": s["reviews"],
                "distance_from_selected_km": 0.0,
                "similarity_match_percent": 90.0,
                "reference_spot": None
            })

        return candidates

