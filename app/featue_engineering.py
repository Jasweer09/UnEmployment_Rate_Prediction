import pandas as pd
from sklearn.impute import SimpleImputer

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
    # Ensure 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convert to datetime, handle errors
    
    # Check for null values after conversion
    if df['date'].isnull().any():
        print("❌ Warning: There are invalid date values after conversion.")
    
    # Convert date to number of days since the first date in the dataset
    df['date'] = (df['date'] - df['date'].min()).dt.days
    
    # Add month, year, and quarter as additional features
    df['month'] = df['date'].apply(lambda x: pd.Timestamp(x).month)
    df['year'] = df['date'].apply(lambda x: pd.Timestamp(x).year)
    df['quarter'] = df['date'].apply(lambda x: pd.Timestamp(x).quarter)
    
    return df


def prepare_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    df = df.sort_values('date').reset_index(drop=True)
    print("Sorting compled")
    #df = add_lag_features(df, feature_cols)
    print("added lag features compled")
    #df = add_rolling_features(df, feature_cols)
    print("added rolling features compled")
    df = add_date_parts(df)
    print("added date compled")
    print("df size", df.shape, df.isna().sum())
    # Impute missing values before training or prediction
    imputer = SimpleImputer(strategy='mean')  # Can also use 'median' or other strategies
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    print("df size after imputer", df.shape)
    print('prepare feature completed')
    return df
