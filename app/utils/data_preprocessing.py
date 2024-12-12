import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

REQUIRED_COLUMNS_MAP = {
    'Demographic': {
        'columns': {
            'Age': {'type': 'int', 'range': '18-80', 'example': 35},
            'Gender': {'type': 'str', 'example': 'Male/Female'},
            'Marital Status': {'type': 'str', 'example': 'Married/Single'},
            'Education Level': {'type': 'str', 'example': "Bachelor's/Master's"}
        }
    },
    'Geographic': {
        'columns': {
            'Country': {'type': 'str', 'example': 'Germany'},
            'City': {'type': 'str', 'example': 'Berlin'},
            'Region': {'type': 'str', 'example': 'Brandenburg'}
        }
    },
    'Behavioral': {
        'columns': {
            'Behavioral Data': {'type': 'str', 'example': 'High Spend'},
            'Purchase History': {'type': 'str', 'example': 'Frequent Purchaser'},
            'Interactions with Customer Service': {'type': 'str', 'example': 'Complaint'}
        }
    },
    'Psychographic': {
        'columns': {
            'Customer Preferences': {'type': 'str', 'example': 'Eco-Friendly'},
            'Preferred Communication Channel': {'type': 'str', 'example': 'Email'},
            'Preferred Contact Time': {'type': 'str', 'example': 'Morning'},
            'Preferred Language': {'type': 'str', 'example': 'English'}
        }
    },
    'Financial': {
        'columns': {
            'Income Level': {'type': 'int', 'range': '10,000-200,000', 'example': 50000},
            'Coverage Amount': {'type': 'int', 'range': '1,000-1,000,000', 'example': 250000},
            'Premium Amount': {'type': 'int', 'range': '100-10,000', 'example': 500}
        }
    },
    'Product-Based': {
        'columns': {
            'Insurance Products Owned': {'type': 'str', 'example': 'Life/Health'},
            'Policy Type': {'type': 'str', 'example': 'Family/Individual'},
            'Customer Preferences': {'type': 'str', 'example': 'Long-Term Coverage'}
        }
    },
    'Engagement-Based': {
        'columns': {
            'Interactions with Customer Service': {'type': 'str', 'example': 'Frequent'},
            'Preferred Communication Channel': {'type': 'str', 'example': 'SMS'}
        }
    }
}

def load_and_prepare_data(segmentation_type, df, column_mapping):
    """
    Load, preprocess, and prepare the data for clustering.
    """
    required_cols = list(REQUIRED_COLUMNS_MAP[segmentation_type]['columns'].keys())
    mapped_columns = {req: column_mapping.get(req) for req in required_cols}  # Map user-selected columns

    # Select only columns that are not "None"
    df_selected = df[[mapped_columns[req] for req in required_cols if mapped_columns[req] is not None]].copy()

    # Encode categorical variables
    for req_col, info in REQUIRED_COLUMNS_MAP[segmentation_type]['columns'].items():
        if info['type'] == 'str' and mapped_columns[req_col] in df_selected:
            le = LabelEncoder()
            df_selected[mapped_columns[req_col]] = le.fit_transform(df_selected[mapped_columns[req_col]].astype(str))

    # Scale numerical columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_selected)

    # Reset index for alignment
    df_selected = df_selected.reset_index(drop=True)
    return scaled_data, df_selected



