# Real-Time E-Commerce Recommender System (`v9-lgbm-hardneg`)

## Overview
This repository contains the production-grade, latency-optimized recommender system developed for the real-time e-commerce recommendation competition. The system combines a **LightGBM** ranking model with robust real-time database synchronization (via TinyBird/ClickHouse) to handle dynamic inventory updates, cold-start items, and user session profiling with strict sub-second SLAs.

---

## Key Features & Architecture

### 1. Hybrid Recommendation Pipeline
- **Candidate Generation (Retrieval):** Combines multiple strategies to surface up to $80$ diverse candidates per request:
  - *Co-visitation matrix* (`co_matrix`) filtered by real-time item availability.
  - *Semantic / Text similarity matching* (TF-IDF/embeddings) for cold-start and rare items.
  - *Category & Subcategory affinity* derived from historical user behavior and live session contexts.
  - *Trending & Popularity tracking* computed over sliding 1-hour event windows (`shop.events`).
- **Scoring & Ranking:** A pre-trained **LightGBM** binary classification/ranking booster evaluates 22 handcrafted features per candidate.
- **Score Calibration:** Optional post-processing grid-search parameters ($\beta, \gamma$) balance pure click-through rates with expected revenue and price sensitivity.

### 2. Real-Time Operational Resilience (`Live-Sync` Patch)
To survive mid-competition catalog updates and sudden stock changes without relying on static `.pkl` snapshots:
- **Positive Inventory Enforcement (`ACTIVE_CATALOG_SQL`):** Bypasses static snapshots by querying `shop.catalog` directly in real-time (`WHERE is_active = 1`).
- **Dynamic New Product Injection:** Newly introduced products are automatically detected, assigned smoothed baseline conversation/cart rates, and prioritized in their respective subcategory pools instantly.
- **Strict Output Validation (`_validated_top10`):** Enforces a final sanity check right before submission to guarantee exactly 10 unique, active product IDs per request, reducing feedback rejection rates (`fb_rejected`) to zero.

---

## Feature Engineering ($22$ Features)
The model computes features entirely in local memory during batch inference without network overhead:
1. **Historical Performance:** `purchases`, `conv_rate`, `cart_rate`, `recent_popularity`.
2. **Pricing Context:** `price`, `price_ratio` (relative to session history or user reference price).
3. **User-Item Interaction & Affinity:** `same_subcat`, `same_cat`, `same_brand`, `user_subcat_share`, `user_brand_share`, `user_past_item_status`.
4. **Co-occurrence & Temporal Signals:** `co_score_last`, `co_score_total`, `purchases_24h`, `views_24h`, `carts_24h`, `product_age_days`.
5. **Contextual Alignment:** `ctx_subcat_affinity` (device/region matching).

---

## Project Structure
```text
starter/
│
├── recommender5.py       # Production Champion (v9-lgbm-hardneg + live catalog sync)
├── recommender6.py       # Experimental Variant (TF-IDF text similarity & calibration)
├── training_boosting5.py # Temporal split offline training pipeline with hard negatives
└── models/
    ├── modelo_lgbm9.pkl  # Trained LightGBM booster weights (v9)
    ├── features_data5.pkl# Serialized catalog, co-matrices, and stats dictionaries
    └── training_report5.json # Offline ablation and validation metrics report
```

---

## Quick Start & Deployment

1. **Environment Setup:**
   Ensure your `.env` file contains the required API credentials:
   ```env
   RAWTREE_API_URL=https://api.rawtree.com
   RAWTREE_API_KEY=your_api_key_here
   TEAM_ID=team_31
   ```

2. **Run the Production Champion:**
   Kill any competing local Python processes to avoid split-brain submissions, then launch the live-synced recommender:
   ```powershell
   python starter/recommender5.py
   ```

---

## Performance & Metrics Summary
- **Validation Strategy:** Chronological session split (preventing data leakage).
- **Primary Objectives:** Maximizing `@10` and `@3` hit rates, purchase hit rate, and expected revenue per request ($\text{EUR/req}$) while maintaining an inference latency well below the strict $3000\text{ms}$ SLA.
