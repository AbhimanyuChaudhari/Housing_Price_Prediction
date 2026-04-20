"""
preprocess.py
-------------
Reusable preprocessing pipeline for the Ames Housing dataset.
Handles missing values, ordinal encoding, skewness correction, and scaling.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler, LabelEncoder


# ── Ordinal mappings ──────────────────────────────────────────────────────────

QUALITY_MAP = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

ORDINAL_COLS = [
    'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
    'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC',
]

# Categorical NAs that mean "feature does not exist"
NONE_COLS = [
    'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
    'MasVnrType',
]

# Numeric NAs that mean "feature does not exist" → 0
ZERO_COLS = [
    'GarageYrBlt', 'GarageArea', 'GarageCars',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
    'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea',
]


# ── Missing value imputation ──────────────────────────────────────────────────

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using domain knowledge about the Ames dataset."""
    df = df.copy()

    for col in NONE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna('None')

    for col in ZERO_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # LotFrontage: impute by neighborhood median (more accurate than global)
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median())
        )

    # MSSubClass is numeric but is actually a category
    if 'MSSubClass' in df.columns:
        df['MSSubClass'] = df['MSSubClass'].astype(str)

    # Remaining: mode for categoricals, median for numerics
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

    return df


# ── Encoding ──────────────────────────────────────────────────────────────────

def encode_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ordinal encoding to quality/condition columns."""
    df = df.copy()
    for col in ORDINAL_COLS:
        if col in df.columns:
            df[col] = df[col].map(QUALITY_MAP).fillna(0).astype(int)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode remaining object columns."""
    df = df.copy()
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


# ── Skewness correction ───────────────────────────────────────────────────────

def fix_skewness(df: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
    """Log1p-transform numeric features with |skew| > threshold."""
    df = df.copy()
    numeric = df.select_dtypes(include=[np.number]).columns
    skewed = df[numeric].apply(lambda x: stats.skew(x.dropna()))
    skewed_cols = skewed[abs(skewed) > threshold].index
    for col in skewed_cols:
        df[col] = np.log1p(df[col].clip(lower=0))
    print(f'  Log-transformed {len(skewed_cols)} skewed features (threshold={threshold})')
    return df


# ── Full pipeline ─────────────────────────────────────────────────────────────

def full_pipeline(train: pd.DataFrame, test: pd.DataFrame):
    """
    Run the complete preprocessing pipeline on combined train+test data.

    Returns
    -------
    X_train, X_test : np.ndarray  (scaled)
    feature_names   : list[str]
    scaler          : fitted RobustScaler
    """
    from src.features import engineer_features

    n_train = len(train)
    y = np.log1p(train['SalePrice'])

    all_data = pd.concat(
        [train.drop(['SalePrice', 'Id'], axis=1, errors='ignore'),
         test.drop('Id', axis=1, errors='ignore')],
        axis=0
    ).reset_index(drop=True)

    print('Step 1/5: Imputing missing values...')
    all_data = impute_missing(all_data)

    print('Step 2/5: Engineering features...')
    all_data = engineer_features(all_data)

    print('Step 3/5: Encoding ordinals...')
    all_data = encode_ordinals(all_data)

    print('Step 4/5: Fixing skewness...')
    all_data = fix_skewness(all_data)

    print('Step 5/5: Encoding categoricals & scaling...')
    all_data = encode_categoricals(all_data)

    feature_names = all_data.columns.tolist()
    scaler = RobustScaler()
    X_all = scaler.fit_transform(all_data)

    X_train = X_all[:n_train]
    X_test  = X_all[n_train:]

    print(f'\nDone. X_train: {X_train.shape}, X_test: {X_test.shape}')
    return X_train, X_test, y, feature_names, scaler
