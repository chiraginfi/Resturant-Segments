import pandas as pd
import re
import ast

df = pd.read_csv("/mnt/data/image_recognition/brown_forman_req/new_output/step6_data_score_premium_palce.csv")

# Strip leading/trailing whitespace from column names (source CSV has ' mentions' with a leading space)
df.columns = df.columns.str.strip()
# Strip whitespace from the mentions values themselves
if 'mentions' in df.columns:
    df['mentions'] = df['mentions'].str.strip()

# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def is_true(row, col):
    """Safely handle TRUE/FALSE strings from CSV"""
    return str(row.get(col, "")).strip().upper() == "TRUE"


def safe_num(val, cast=float, default=0):
    """Safely cast a value that may be NaN, None, or empty string."""
    try:
        return cast(val) if pd.notna(val) and str(val).strip() != "" else default
    except (ValueError, TypeError):
        return default


def cost_to_score(cost):
    """
    Mutually exclusive price tiers.
    Based on Mumbai market: ₹600 street food → ₹5000+ fine dining.
    Returns None when cost is missing so fallback layers can trigger.
    """
    if cost is None or (isinstance(cost, float) and pd.isna(cost)):
        return None
    if cost > 4000:   return 8
    if cost > 2500:   return 6
    if cost > 1200:   return 3
    if cost > 600:    return 1
    return 0

# ═══════════════════════════════════════════════════════════
# PRE-COMPUTE: Derived columns added to df before scoring
# ═══════════════════════════════════════════════════════════

DRINK_PRICE_AVG_COLS = [
    'beer_item_price_avg', 'gin_item_price_avg', 'whisky_item_price_avg',
    'vodka_item_price_avg', 'rum_item_price_avg', 'tequila_item_price_avg',
    'martini_item_price_avg', 'wine_item_price_avg', 'mojito_item_price_avg',
    'other_item_price_avg', 'spirits_item_price_avg', 'other_cocktails_item_price_avg',
    'mezcal_item_price_avg', 'soju_item_price_avg', 'picante_item_price_avg',
    'absinthe_item_price_avg', 'brandy_item_price_avg', 'old_fashioned_item_price_avg'
]

DRINK_COUNT_COLS = [c.replace('_item_price_avg', '_item_count') for c in DRINK_PRICE_AVG_COLS]
BRAND_NAME_COLS  = [c.replace('_item_price_avg', '_brand_name') for c in DRINK_PRICE_AVG_COLS]

# Max avg drink price across all categories
def _max_drink_price(row):
    prices = [safe_num(row.get(c)) for c in DRINK_PRICE_AVG_COLS if pd.notna(row.get(c))]
    return max(prices) if prices else 0

df['max_drink_price_avg'] = df.apply(_max_drink_price, axis=1)

# Premium spirit flag: whisky > ₹800 OR wine > ₹700 OR cocktails > ₹600
df['premium_spirit_flag'] = (
    (df.get('whisky_item_price_avg', 0).fillna(0) > 800) |
    (df.get('wine_item_price_avg',   0).fillna(0) > 700) |
    (df.get('other_cocktails_item_price_avg', 0).fillna(0) > 600)
)

# Total drink items across all categories
existing_count_cols = [c for c in DRINK_COUNT_COLS if c in df.columns]
df['total_item_count'] = df[existing_count_cols].fillna(0).sum(axis=1).astype(int)

# Number of distinct drink categories present (brand_name not null/other)
def _drink_category_count(row):
    return sum(
        1 for c in BRAND_NAME_COLS
        if pd.notna(row.get(c)) and str(row.get(c)).strip().lower() not in ("", "nan", "other")
    )
df['drink_category_count'] = df.apply(_drink_category_count, axis=1)

# ═══════════════════════════════════════════════════════════
# PROXY 1 — Infer spend level from drink/brand item prices
# ═══════════════════════════════════════════════════════════

def infer_cost_from_drinks(row):
    """
    Median drink price × 6 ≈ implied cost for two.
    Rationale: alcohol typically 15–20% of total bill.
    A cocktail at ₹800+ → premium venue; ₹250 → casual bar.
    Receives -1 score haircut downstream (inferred, not observed).
    """
    prices = []
    for col in DRINK_PRICE_AVG_COLS:
        val = row.get(col)
        if pd.notna(val) and str(val).strip().lower() not in ("", "nan", "other"):
            try:
                prices.append(float(str(val).replace(",", "")))
            except ValueError:
                pass
    if not prices:
        return None
    median_drink = sorted(prices)[len(prices) // 2]
    return median_drink * 6

premium_brands = {
    'Goenchi','Cazulo','Moji','Volando','Aani Ek','Fidalgo',
    'Camikara','Maka Zai','Segredo Aldeia','Short Story',
    'Corona','Hoegaarden','Heineken','Bira 91','Simba','Budweiser','Carlsberg',
    'Kingfisher Ultra','Kingfisher Storm','Stella Artois','Erdinger',
    'Remy Martin','Hennessy','Martell','Courvoisier','Paul John','St-Rémy',
    'Morpheus','Roulette',
    'Grey Goose','Belvedere','Ciroc','Smoke','Absolut','Ketel One','Skyy',
    'Hapusa','Jaisalmer','Stranger & Sons','Greater Than','Hendrick’s','Hendricks',
    'Roku','Tanqueray','Bombay Sapphire','Samsara','Terai',
    'Amrut','Indri','Rampur','Godawan','Chivas Regal','The Singleton',
    'Glenfiddich','The Glenlivet','Jameson','Jack Daniels','Black Dog',
    '100 Pipers','Teacher’s','Ballantine’s','Wood Burns'
}

# ═══════════════════════════════════════════════════════════
# PROXY 2 — Review text mining
# Three signal types + one negative/penalty type
# ═══════════════════════════════════════════════════════════

# (+) Price keywords: explicit spend mentions
PRICE_KW = [
    '₹2,000', '₹3,000', '₹4,000', '₹5,000',
    '₹2000',  '₹3000',  '₹4000',  '₹5000',
    'pricey', 'premium', 'expensive', 'steep', 'splurge',
    'worth every penny', 'high end', 'high-end', 'not cheap',
    'costs a lot', 'on the pricier side', 'pocket pinch'
]

# (+) Experience/quality keywords
EXPERIENCE_KW = [
    'fine dining', 'upscale', 'luxur', 'exclusive', 'world class',
    'top notch', 'exceptional', 'impeccable', 'michelin',
    'sophisticated', 'elevated', 'white glove', 'curated menu',
    'chef', 'sommelier', 'tasting menu'
]

# (+) Crowd/vibe keywords
VIBE_KW = [
    'rooftop', 'intimate', 'romantic', 'buzzing', 'lively',
    'date night', 'special occasion', 'anniversary', 'celebration',
    'chic', 'trendy', 'vibrant', 'ambience', 'atmosphere',
    'stunning view', 'beautiful decor', 'instagrammable'
]

# (−) Negative price shock: bad value signals → subtract score
NEGATIVE_KW = [
    'overpriced', 'not worth', 'too expensive', 'rip off', 'ripoff',
    'over priced', 'not worth the price', 'not worth the hype',
    'daylight robbery', 'exorbitant', 'highway robbery',
    'waste of money', 'poor value', 'not worth it'
]

def mine_reviews(row):
    """
    Parse the raw reviews list, scan for all four keyword buckets.
    Returns: (price_hits, experience_hits, vibe_hits, negative_hits)
    """
    raw = row.get("reviews", "[]")
    price_hits = experience_hits = vibe_hits = negative_hits = 0
    try:
        reviews = ast.literal_eval(raw) if isinstance(raw, str) else (raw or [])
        full_text = " ".join(str(r) for r in reviews).lower()

        price_hits      = sum(1 for kw in PRICE_KW      if kw.lower() in full_text)
        experience_hits = sum(1 for kw in EXPERIENCE_KW if kw.lower() in full_text)
        vibe_hits       = sum(1 for kw in VIBE_KW        if kw.lower() in full_text)
        negative_hits   = sum(1 for kw in NEGATIVE_KW    if kw.lower() in full_text)
    except Exception:
        pass
    return price_hits, experience_hits, vibe_hits, negative_hits

# ═══════════════════════════════════════════════════════════
# MASTER CLASSIFIER
# ═══════════════════════════════════════════════════════════

def classify_restaurant(row):
    score = 0
    breakdown = []
    price_source = "direct"

    # ── LAYER 1: Direct cost_for_two (menu-stated) ────────
    cost = row.get("cost_for_two_food")
    price_score = cost_to_score(cost)

    # ── LAYER 2: Fallback → cost_for_two_drinks (observed drink prices × 4) ──
    # More reliable than median-inferred because it uses actual avg prices from
    # the menu; receives a -1 haircut since it covers drinks only, not full bill.
    if price_score is None:
        drinks_cost = row.get("cost_for_two_drinks")
        price_score = cost_to_score(drinks_cost)
        price_source = "drinks_cost_observed"
        if price_score is not None:
            price_score = max(0, price_score - 1)   # haircut: drinks-only cost

    if price_score is None:
        dining_cost = row.get("dining_price")
        price_score = cost_to_score(dining_cost)
        price_source = "dining_cost_observed"
        if price_score is not None:
            price_score = max(0,price_score-2)
    # ── LAYER 3: Fallback → infer from median drink price ─
    if price_score is None:
        implied_cost = infer_cost_from_drinks(row)
        price_score  = cost_to_score(implied_cost)
        price_source = "drinks_inferred"
        if price_score is not None:
            price_score = max(0, price_score - 3)   # larger haircut: estimated cost

    # ── LAYER 4: Fallback → bootstrap from reviews alone ──
    price_hits, experience_hits, vibe_hits, negative_hits = mine_reviews(row)

    if price_score is None:
        price_score  = min(price_hits, 2) + min(experience_hits, 2)
        price_source = "reviews_inferred"

    ps = price_score if price_score is not None else 0
    score += ps
    # Always show price in breakdown so Budget-tier (score=0) is visible too
    breakdown.append(f"price[{price_source}]:+{ps}")

    # ── Review signals (additive regardless of price source) ──
    exp_pts  = min(experience_hits, 2)
    vibe_pts = min(vibe_hits, 2)
    price_kw_pts = min(price_hits, 1)
    neg_pts  = min(negative_hits * 2, 4)

    score += exp_pts
    score += vibe_pts
    score += price_kw_pts
    score -= neg_pts

    if exp_pts:      breakdown.append(f"review_experience:+{exp_pts}")
    if vibe_pts:     breakdown.append(f"review_vibe:+{vibe_pts}")
    if price_kw_pts: breakdown.append(f"review_price_kw:+{price_kw_pts}")
    if neg_pts:      breakdown.append(f"neg_review_penalty:-{neg_pts}")

    # ── Structural flags ──────────────────────────────────
    flag_map = {
        "atmosphere__feels_upscale":                    2,
        "atmosphere__feels_romantic":                   1,
        "planning__requires_reservations":              1,
        "planning__recommends_reservations_dinner":     1,
        "planning__recommends_reservations_lunch":      1,
        "planning__recommends_reservations_brunch":     1,
        "parking__has_parking_valet":                   1,
        "offerings__has_private_dining_room":           2,
        "highlights__has_live_music":                   1,
        "highlights__has_seating_rooftop":              1,
        "offerings__serves_cocktails":                  1,
        "offerings__serves_wine":                       1,
        "atmosphere__feels_hip":                        1,
        "offerings__serves_happy_hour_drinks_x":        1,
        "amenities__has_bar_onsite":                    1,
        "offerings__serves_late_night_food":            1,
        "offerings__has_dancing":                       1,
        "atmosphere__is_recently_popular":              1,
        "highlights__has_bar_games":                    1,
        "highlights__has_karaoke_nights":               1,
        "atmosphere__feels_casual":                    -1,   # casual = lower tier
    }
    for col, pts in flag_map.items():
        if is_true(row, col):
            score += pts
            if pts > 0:
                breakdown.append(f"{col.split('__')[-1]}:+{pts}")
            else:
                breakdown.append(f"{col.split('__')[-1]}:{pts}")

    # ── Premium spirit pricing ─────────────────────────────
    premium_flag = row.get("premium_spirit_flag", False)
    max_drink    = safe_num(row.get("max_drink_price_avg"))
    if premium_flag:
        prem_pts = 2 if max_drink > 1200 else 1
        score += prem_pts
        breakdown.append(f"premium_spirit:+{prem_pts}")

    # ── Drink menu depth (breadth = sophistication proxy) ──
    drink_depth = safe_num(row.get("drink_category_count"), cast=int)
    if drink_depth > 6:
        depth_pts = 3
    elif drink_depth > 3:
        depth_pts = 2
    elif drink_depth > 0:
        depth_pts = 1
    else:
        depth_pts = 0
    score += depth_pts
    if depth_pts:
        breakdown.append(f"drink_category_depth({drink_depth}_cats):+{depth_pts}")

    # ── Total item count (menu comprehensiveness) ──────────
    total_items = safe_num(row.get("total_item_count"), cast=int)
    if total_items > 50:
        item_pts = 2
    elif total_items > 20:
        item_pts = 1
    else:
        item_pts = 0
    score += item_pts
    if item_pts:
        breakdown.append(f"menu_volume({total_items}_items):+{item_pts}")

    # ── Ratings ───────────────────────────────────────────
    rating = safe_num(row.get("ratings"))
    if rating >= 4.5:
        rating_pts = 2
    elif rating >= 4.0:
        rating_pts = 1
    else:
        rating_pts = -1
    score += rating_pts
    if rating_pts > 0:  breakdown.append(f"rating({rating}):+{rating_pts}")
    if rating_pts < 0:  breakdown.append(f"rating({rating}):{rating_pts}")

    # ── Review volume (popularity signal) ─────────────────
    reviews_count = safe_num(row.get("reviews_count"))
    if reviews_count > 3000:
        vol_pts = 2
    elif reviews_count > 1000:
        vol_pts = 1
    else:
        vol_pts = 0
    score += vol_pts
    if vol_pts:
        breakdown.append(f"review_volume({int(reviews_count)}):+{vol_pts}")

    # ── Mentions (external brand citation) ────────────────
    mentions_val = row.get("mentions")
    has_mention  = pd.notna(mentions_val) and str(mentions_val).strip() not in ("", "nan")
    if has_mention:
        score += 3
        breakdown.append("mentions:+3")

    # ── Segment thresholds ────────────────────────────────
    # Calibrated against your 4 sample restaurants:
    # O Pedro (BKC): expect Luxury    ~score 18+
    # Bandra Born:   expect Luxury    ~score 16+
    # La Loca Maria: expect Premium   ~score 13+
    # Mizu:          expect Premium   ~score 12+
    # Veronica's:    expect Budget    ~score 4
    if score >= 16:   segment = "Luxury"
    elif score >= 10: segment = "Premium"
    elif score >= 5:  segment = "Mid"
    else:             segment = "Budget"

    return pd.Series({
        "restaurant_segment":   segment,
        "segment_score":        score,
        "segment_price_source": price_source,
        "score_breakdown":      " | ".join(breakdown),
    })


df[["restaurant_segment", "segment_score", "segment_price_source", "score_breakdown"]] = df.apply(
    classify_restaurant, axis=1
)

import ast

df['brand_count_dict'] = df['brand_count_dict'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

def count_premium(d):
    if not isinstance(d, dict):
        return 0
    
    total = 0
    for brand, count in d.items():
        brand_clean = brand.replace("’", "'").strip()
        
        # normalize Hendrick's issue
        if brand_clean.lower() in ["hendrick’s", "hendricks"]:
            brand_clean = "Hendrick’s"
        
        if brand_clean in premium_brands:
            total += count
    
    return total

df['premium_brand_count'] = df['brand_count_dict'].apply(count_premium)

dummies = pd.get_dummies(df["mentions"]).astype(int)
df_final = df.join(dummies)

df_final = df_final.drop_duplicates(subset=["poi_code"])
df_final.to_csv(
    "/mnt/data/image_recognition/brown_forman_req/new_output/step7_restaurant__segmented.csv",
    index=False
)
print("Done. Shape:", df.shape)
print(df[["name", "segment_score", "restaurant_segment", "score_breakdown"]].head(10).to_string(index=False))
