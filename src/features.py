"""
features.py
-----------
All feature engineering for the Ames Housing dataset.
Each function is documented with the rationale for the feature.
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['TotalSF'] = (
        df.get('TotalBsmtSF', 0) +
        df.get('1stFlrSF', 0) +
        df.get('2ndFlrSF', 0)
    )
    df['TotalLivingArea'] = (
        df.get('GrLivArea', 0) +
        df.get('TotalBsmtSF', 0)
    )
    df['TotalPorchSF'] = (
        df.get('OpenPorchSF', 0) +
        df.get('EnclosedPorch', 0) +
        df.get('3SsnPorch', 0) +
        df.get('ScreenPorch', 0) +
        df.get('WoodDeckSF', 0)
    )
    df['TotalBathrooms'] = (
        df.get('FullBath', 0) +
        0.5 * df.get('HalfBath', 0) +
        df.get('BsmtFullBath', 0) +
        0.5 * df.get('BsmtHalfBath', 0)
    )
    df['HouseAge']    = df.get('YrSold', 2010) - df.get('YearBuilt', 2000)
    df['RemodelAge']  = df.get('YrSold', 2010) - df.get('YearRemodAdd', 2000)
    df['GarageAge']   = df.get('YrSold', 2010) - df.get('GarageYrBlt', 2000)
    df['IsNewHouse']  = (df.get('YearBuilt', 0) == df.get('YrSold', 1)).astype(int)
    df['IsRemodeled'] = (df.get('YearBuilt', 0) != df.get('YearRemodAdd', 0)).astype(int)

    df['HasGarage']    = (df.get('GarageArea', 0) > 0).astype(int)
    df['HasPool']      = (df.get('PoolArea', 0) > 0).astype(int)
    df['HasFireplace'] = (df.get('Fireplaces', 0) > 0).astype(int)
    df['HasBsmt']      = (df.get('TotalBsmtSF', 0) > 0).astype(int)
    df['Has2ndFloor']  = (df.get('2ndFlrSF', 0) > 0).astype(int)
    df['HasPorch']     = (df.get('TotalPorchSF', 0) > 0).astype(int)

    if 'OverallQual' in df.columns:
        df['QualxSF']      = df['OverallQual'] * df['TotalSF']
        df['QualxLivArea'] = df['OverallQual'] * df.get('GrLivArea', 0)
    if 'GarageArea' in df.columns and 'GarageCars' in df.columns:
        df['GarageScore']  = df['GarageArea'] * df['GarageCars']

    expensive_hoods = ['NridgHt','NoRidge','StoneBr','Timber','Veenker','Somerst','ClearCr','Crawfor']
    mid_hoods = ['CollgCr','Blmngtn','Gilbert','NWAmes','SawyerW','Mitchel','NAmes','NPkVill']
    if 'Neighborhood' in df.columns:
        df['NeighborhoodTier'] = df['Neighborhood'].apply(
            lambda x: 2 if x in expensive_hoods else (1 if x in mid_hoods else 0)
        )

    return df
