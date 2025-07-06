import pandas as pd

def add_lag_features(df: pd.DataFrame, columns: list, lags: int = 4) -> pd.DataFrame:
    for col in columns:
        for lag in range(1, lags + 1):
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df

def add_rolling_features(df: pd.DataFrame, columns: list, windows: list = [3, 6]) -> pd.DataFrame:
    for col in columns:
        for win in windows:
            df[f"{col}_rolling_mean_{win}"] = df[col].rolling(window=win).mean()
            df[f"{col}_rolling_std_{win}"] = df[col].rolling(window=win).std()
    return df

def add_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    return df

def prepare_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    df = df.sort_values('date').reset_index(drop=True)
    df = add_lag_features(df, feature_cols)
    df = add_rolling_features(df, feature_cols)
    df = add_date_parts(df)
    return df
