import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Define the required columns for each segmentation type
REQUIRED_COLUMNS_MAP = {
    'Demographic': ["Age", "Gender", "Income Level", "Education Level", "Occupation", "Marital Status",
                    "Insurance Products Owned"],
    'Geographic': ["Country", "City", "Region"],
    'Behavioral': ["Purchase Frequency", "Average Spend", "Loyalty Score"],
    'Psychographic': ["Lifestyle", "Interests", "Values"],
    'ProductUsage': ["Product Category", "Usage Frequency", "Last Purchase Days Ago"]
}


def load_and_prepare_data(segmentation_type, df, column_mapping):
    # Apply column mapping
    rename_map = {v: k for k, v in column_mapping.items() if v is not None}
    df = df.rename(columns=rename_map)

    # Ensure required columns are present
    required_cols = REQUIRED_COLUMNS_MAP[segmentation_type]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Select and preprocess the required columns
    df_selected = df[required_cols].copy()

    # Encode categorical variables
    for col in df_selected.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_selected[col] = le.fit_transform(df_selected[col].astype(str))

    # Scale numerical features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_selected)

    # Reset index for alignment
    df_selected.reset_index(drop=True, inplace=True)
    return scaled_data, df_selected

