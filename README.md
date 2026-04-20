# Housing Price Regression

A research-grade regression project on the [Kaggle Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques), built from scratch with a focus on understanding *why* prices are what they are not just achieving a score.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-tuned-orange?style=flat-square)
![LightGBM](https://img.shields.io/badge/LightGBM-tuned-green?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-explainability-purple?style=flat-square)
![Optuna](https://img.shields.io/badge/Optuna-hyperparameter--tuning-red?style=flat-square)

---

## Results

| Model | CV RMSE (log) | Notes |
|---|---|---|
| Ridge (baseline) | ~0.1423 | Strong with log-transform |
| Lasso | ~0.1441 | Useful for feature selection |
| ElasticNet | ~0.1456 | - |
| SVR | ~0.1380 | Slow, competitive |
| Random Forest | ~0.1387 | Good baseline |
| Gradient Boosting | ~0.1262 | Solid |
| **XGBoost (tuned)** | **~0.1158** | Best single model |
| **LightGBM (tuned)** | **~0.1165** | Comparable to XGBoost |
| **Stacking Ensemble** | **~0.1132** | Best overall |

> RMSE is in log(SalePrice) space - 0.115 means ~±11.5% error on average.

---

## What Makes This Research-Grade

This isn't a tutorial notebook. It's a structured investigation:

**EDA first, model second.** Nine visualizations before a single model is trained - target distribution, missing value audit, outlier detection, correlation analysis, neighborhood deep-dive, seasonality trends.

**Feature engineering with rationale.** 15 new features created, each with documented reasoning. `TotalSF` (all floor areas combined) consistently ranks in the top 3 features by SHAP importance — beating any individual floor area.

**Optuna hyperparameter tuning.** 50-trial Bayesian optimization for XGBoost and LightGBM, producing meaningful improvements over default parameters.

**Stacking ensemble.** Two-level stacking with Ridge, Lasso, GBM, XGBoost, LightGBM as base learners and Ridge as meta-learner. Cross-validated to prevent leakage.

**SHAP explainability.** Global importance, beeswarm plots, dependence plots, and individual prediction waterfall charts for 3 representative houses.

**Error analysis.** Not just R² - where does the model fail by price range, quality tier, and decade built? (Answer: luxury homes >$400K have systematically higher errors.)

---

## Project Structure

```
Housing-Price-Regression/
├── data/
│   ├── train.csv                   # Ames Housing training data (1,460 rows)
│   ├── test.csv                    # Test set
│   ├── submission.csv              # Final predictions (generated)
│   └── processed/                  # Pickled pipeline data (generated)
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis (9 plots)
│   ├── 02_preprocessing.ipynb      # Feature engineering & preprocessing (5 plots)
│   ├── 03_modeling.ipynb           # Model training, tuning, ensemble (5 plots)
│   └── 04_explainability.ipynb     # SHAP, error analysis, what-if (8 plots)
├── src/
│   ├── preprocess.py               # Reusable preprocessing pipeline
│   ├── features.py                 # Feature engineering functions
│   └── models.py                   # Model definitions, Optuna tuning, stacking
├── plots/                          # All generated visualizations (generated)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Notebooks

### `01_eda.ipynb` - Exploratory Data Analysis
- Target variable distribution + normality tests (D'Agostino-Pearson)
- Missing value audit with domain interpretation
- Skewness analysis across all 79 features
- Outlier detection (GrLivArea vs SalePrice scatter)
- Correlation analysis + heatmap for top 15 features
- Categorical feature analysis (OverallQual, Neighborhood, MSZoning, etc.)
- Neighborhood deep-dive: price tier + variability (CV) per neighborhood
- Time trends: price by decade built, seasonality by month

### `02_preprocessing.ipynb` - Feature Engineering & Preprocessing
- Outlier removal (2 partial-sale transactions)
- Domain-aware missing value imputation (most NAs = "no feature", not unknown)
- `LotFrontage` imputed by neighborhood median (not global)
- 15 engineered features: area aggregations, age features, binary flags, quality×area interactions, neighborhood tier
- Validation: all engineered features plotted against SalePrice to confirm they add signal
- Ordinal encoding for 10 quality/condition columns
- Skewness correction with `log1p` (75+ features transformed)
- `RobustScaler` (chosen over StandardScaler for outlier robustness)

### `03_modeling.ipynb` - Training & Ensemble
- 5-fold CV comparison across 8 baseline models
- Optuna Bayesian optimization (50 trials each for XGBoost + LightGBM)
- Stacking ensemble: 5 base learners → Ridge meta-learner with `passthrough=True`
- Learning curves: diagnose bias/variance tradeoff
- Weighted blend (XGB 30% + LGB 30% + Stack 25% + GBM 10% + Ridge 5%)
- Prediction distribution comparison across models

### `04_explainability.ipynb` - SHAP & Error Analysis
- SHAP TreeExplainer: global importance (top 25 features)
- SHAP beeswarm: direction + magnitude for all features simultaneously
- SHAP dependence plots: non-linear relationships for top 4 features
- Individual waterfall explanations for cheap / median / expensive houses
- Error analysis: actual vs predicted, residuals, cumulative error distribution
- Error by group: price range, OverallQual, decade built
- Permutation importance: model-agnostic validation of feature rankings
- What-if analysis: how does predicted price change when you vary OverallQual or TotalSF?

---

## Key Findings

**Feature engineering > model selection.** The jump from Ridge (0.142) to tuned XGBoost (0.116) is partly the model, but `TotalSF`, `QualxSF`, and `NeighborhoodTier` - all engineered features — consistently appear in the top 10 by both SHAP and permutation importance.

**OverallQual is non-linear.** SHAP dependence plots reveal that each quality grade adds increasingly more value at the top end — going from grade 8 to 9 adds more than going from grade 5 to 6.

**Luxury homes are hardest to price.** Error analysis shows homes >$400K have ~2× the error rate of mid-range homes. These properties have unique features not well-represented in the training data (small sample, high variance).

**Most "missing" values aren't missing.** 60%+ of NAs in this dataset mean "the feature doesn't exist" (no pool, no fence, no alley). Treating them as truly missing and imputing with the mean would introduce noise.

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/AbhimanyuChaudhari/Housing-Price-Regression.git
cd Housing-Price-Regression

# Install dependencies
pip install -r requirements.txt

# Download data from Kaggle
# Place train.csv and test.csv in data/

# Run notebooks in order
jupyter lab  # or open in VS Code
```

Run the notebooks **in order** (01 → 02 → 03 → 04). Each notebook saves outputs used by the next.

---

## Tech Stack

- **Data:** pandas, NumPy, SciPy
- **Modeling:** scikit-learn, XGBoost, LightGBM
- **Tuning:** Optuna (Bayesian optimization)
- **Explainability:** SHAP
- **Visualization:** Matplotlib, Seaborn

---

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

**Abhimanyu Chaudhari** — MS Financial Technologies, NJIT
[LinkedIn](http://www.linkedin.com/in/abhimanyu-chaudhari16) · [GitHub](https://github.com/AbhimanyuChaudhari) · abhimanyuchaudhari16@gmail.com
