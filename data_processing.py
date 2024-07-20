# src/utils/data_processing.py

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from a CSV file."""
        return pd.read_csv(file_path)

    def load_json(self, file_path: str) -> Dict[str, Any]:
        """Load data from a JSON file."""
        return pd.read_json(file_path).to_dict()

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic data cleaning: remove duplicates and handle missing values."""
        df = df.drop_duplicates()
        df = df.fillna(df.mean())
        return df

    def normalize_numeric_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Normalize numeric features using StandardScaler."""
        df[columns] = self.scaler.fit_transform(df[columns])
        return df

    def encode_categorical_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Encode categorical features using LabelEncoder."""
        for col in columns:
            df[col] = self.label_encoder.fit_transform(df[col])
        return df

    def impute_missing_values(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Impute missing values using SimpleImputer."""
        df[columns] = self.imputer.fit_transform(df[columns])
        return df

    def split_data(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2):
        """Split data into training and testing sets."""
        from sklearn.model_selection import train_test_split
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=42)

    def create_time_series_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Create time series features from a date column."""
        df[date_column] = pd.to_datetime(df[date_column])
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['day_of_week'] = df[date_column].dt.dayofweek
        return df

    def aggregate_data(self, df: pd.DataFrame, group_by: List[str], agg_dict: Dict[str, List[str]]) -> pd.DataFrame:
        """Aggregate data based on specified columns and functions."""
        return df.groupby(group_by).agg(agg_dict).reset_index()

    def detect_outliers(self, df: pd.DataFrame, column: str, threshold: float = 3) -> pd.Series:
        """Detect outliers using the Z-score method."""
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return z_scores > threshold

    def one_hot_encode(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Perform one-hot encoding on specified categorical columns."""
        return pd.get_dummies(df, columns=columns)

    def scale_features(self, df: pd.DataFrame, columns: List[str], feature_range: tuple = (0, 1)) -> pd.DataFrame:
        """Scale features to a specified range."""
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=feature_range)
        df[columns] = scaler.fit_transform(df[columns])
        return df

    def save_to_csv(self, df: pd.DataFrame, file_path: str) -> None:
        """Save DataFrame to a CSV file."""
        df.to_csv(file_path, index=False)

    def save_to_json(self, data: Dict[str, Any], file_path: str) -> None:
        """Save dictionary to a JSON file."""
        import json
        with open(file_path, 'w') as f:
            json.dump(data, f)

# Example usage
if __name__ == "__main__":
    processor = DataProcessor()
    
    # Load data
    df = processor.load_csv("example_data.csv")
    
    # Clean and preprocess
    df = processor.clean_data(df)
    df = processor.normalize_numeric_features(df, ['age', 'salary'])
    df = processor.encode_categorical_features(df, ['gender', 'department'])
    
    # Create time series features
    df = processor.create_time_series_features(df, 'date')
    
    # Detect outliers
    outliers = processor.detect_outliers(df, 'salary')
    
    # Save processed data
    processor.save_to_csv(df, "processed_data.csv")
    
    print("Data processing complete!")