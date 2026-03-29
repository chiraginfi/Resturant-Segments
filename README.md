# POI data pipeline (steps 1–10)

This folder contains sequential Python scripts that build a single enriched restaurant / POI dataset for Brown‑Forman analytics. Paths are **hard‑coded** under `brown_forman_req/new_input/` (inputs) and `brown_forman_req/new_output/` (intermediate and final CSVs).

Run steps **in order** unless you are restarting from a saved artifact.

## Quick reference

| Step | Script | Main output |
|------|--------|-------------|
| 1 | `step1_combine_poi_data.py` | `new_output/step1_poi_data_club.csv` |
| 2 | `step2_zomato_poi_data.py` | `new_output/step2_poi_with_all_features.csv` |
| 3 | `step3_add_cost_for_two.py` | `new_output/step3_include_all.csv` |
| 4 | `step4_removing_true_false.py` | `new_output/restaurant_brand_data_final.csv` |
| 5 | `step5_wbi_poi.py` | `new_output/step5_zomato_poi_wbi.csv` |
| 6 | `step6_weighted_score.py` | `new_output/step6_data_score_premium_palce.csv` |
| 7 | `step7_scroing_sgement.py` | `new_output/step7_restaurant__segmented.csv` |
| 7b | `step7_2_adding_extra.py` | `new_output/step_7_2_output.csv` |
| 8 | `step8_adding_hotel_website.py` | `new_output/step8_data_output.csv` |
| 9 | `step9_add_visits.py` | `new_output/step9_output.csv` |
| 10 | `step10_add_operational.py` | `new_output/step10_output.csv` |

---

## Step 1 — Combine matched and unmatched POI data

**Script:** `step1_combine_poi_data.py`

**Reads:**

- `new_input/input_macthed_poi.csv` (matched POIs)
- `new_input/unmatched_88.csv`
- `new_input/unmatched_927_merged.csv` (for `poi_code` ↔ `url_hash`)

**Does:** Normalizes `poi_code` on the matched file, aligns columns between matched and unmatched rows, then concatenates into one table.

**Writes:** `new_output/step1_poi_data_club.csv`

---

## Step 2 — Zomato / bridge merge and category pivots

**Script:** `step2_zomato_poi_data.py`

**Reads:**

- `new_input/bridge_file.csv`
- `new_input/brownforman_brand_items.csv`
- `new_output/step1_poi_data_club.csv`

**Does:** Joins bridge and brand menu data, attaches POI rows by `url_hash`, builds per‑POI brand count dictionaries and pivoted category columns (avg price, counts, brand names per drink category).

**Writes:** `new_output/step2_poi_with_all_features.csv`

---

## Step 3 — Cost for two, mentions fill‑in, dining price from reviews

**Script:** `step3_add_cost_for_two.py`

**Reads:**

- `new_output/step2_poi_with_all_features.csv`
- `new_input/zomato_cost_for_two.csv`
- `new_input/browformanbars.csv` (mentions)
- `new_input/3kpois.csv` (fills POIs missing from mentions)

**Does:** Merges Zomato “cost for two” text into numeric `cost_for_two_food`, derives `cost_for_two_drinks` from median category drink prices (capped), combines into `cost_for_two_both`, and extracts guided dining price ranges from review text into `dining_price`.

**Writes:** `new_output/step3_include_all.csv`

---

## Step 4 — Merge suffix cleanup and boolean encoding

**Script:** `step4_removing_true_false.py`

**Reads:** `new_output/step3_include_all.csv`

**Does:** Strips column names, collapses `_x` / `_y` duplicate columns from merges, converts string columns that are only `TRUE`/`FALSE` to `1`/`0`.

**Writes:** `new_output/restaurant_brand_data_final.csv`

---

## Step 5 — WBI (wealth / index) by POI

**Script:** `step5_wbi_poi.py`

**Reads:**

- `new_output/restaurant_brand_data_final.csv`
- `new_input/brown_forman_wbi.csv`

**Does:** Normalizes `poi_code`, aggregates `poi_wbi` to median `wbi` per POI, left‑joins onto the main table.

**Writes:** `new_output/step5_zomato_poi_wbi.csv`

---

## Step 6 — Weighted “score” and premium pincode flag

**Script:** `step6_weighted_score.py`

**Reads:** `new_output/step5_zomato_poi_wbi.csv`

**Does:** Builds `cost_final` from food+drinks cost with fallback to `dining_price`, normalizes cost and log‑reviews, computes a weighted `score` from amenities, ratings, reviews, etc., extracts Mumbai‑style `pincode` from `address`, and sets `premium_place` for a fixed set of premium pincodes.

**Writes:** `new_output/step6_data_score_premium_palce.csv`

---

## Step 7 — Restaurant segmentation scoring

**Script:** `step7_scroing_sgement.py` *(filename spelling as in repo)*

**Reads:** `new_output/step6_data_score_premium_palce.csv`

**Does:** Derives drink‑price and assortment features, runs a layered segment classifier (`Luxury` / `Premium` / `Mid` / `Budget`) with `segment_score`, `segment_price_source`, and `score_breakdown`, counts premium brands from `brand_count_dict`, one‑hot encodes `mentions`, drops duplicate `poi_code` rows.

**Writes:** `new_output/step7_restaurant__segmented.csv`

---

## Step 7b — Extra website / hotel / hours columns

**Script:** `step7_2_adding_extra.py`

**Reads:**

- `new_output/step7_restaurant__segmented.csv`
- `new_input/brown_forman_webisite_hotel_stars_of_host_open_close_hours_left_data.csv`

**Does:** Left‑joins additional columns from the “left data” extract keyed on `poi_code` (includes website and open/close hour fields used downstream).

**Writes:** `new_output/step_7_2_output.csv`

---

## Step 8 — Domain validation and Michelin flag

**Script:** `step8_adding_hotel_website.py`

**Reads:**

- `new_output/step_7_2_output.csv`
- `new_input/michlin.csv`

**Does:** Sets `has_domain` from `website_domain_name` using excluded marketplaces / social / short‑link patterns; sets `michelin_star` if `poi_code` appears in the Michelin list.

**Writes:** `new_output/step8_data_output.csv`

---

## Step 9 — Footfall visits (weekday vs weekend)

**Script:** `step9_add_visits.py`

**Reads:**

- `new_output/step8_data_output.csv`
- `new_input/visits_brown_forman.csv`

**Does:** Groups visits by `poi_code` and `day_type`, pivots to `weekday_visits` and `weekend_visits`, merges onto the main table.

**Writes:** `new_output/step9_output.csv`

---

## Step 10 — Operational status from opening hours

**Script:** `step10_add_operational.py`

**Reads:** `new_output/step9_output.csv`

**Does:** Parses `open_close_hours` into timing strings, classifies each POI into `operational_status` (e.g. All‑Day / Evening / Late‑Night / Mixed) based on average start and end times in 24‑hour form.

**Writes:** `new_output/step10_output.csv`

---

## Other scripts (not in the 1–10 chain)

- **`step_test.py`** — Builds `new_input/input_macthed_poi.csv` from `dfpoidata.csv` and `df_h3index.csv` (useful when regenerating step 1 inputs).
- **`delta_diff.py`** — Compares two prediction CSVs under `model_input/` and writes `new_output/delta_diff.csv` for label changes between runs.

---

## Requirements

Scripts use **pandas** (and **numpy** where noted). Ensure those packages are installed in your environment before running.
