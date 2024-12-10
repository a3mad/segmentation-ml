import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_prepare_data(segmentation_type):
    # Load the dataset
    df = pd.read_csv('data/customer_segmentation_data.csv')

    if segmentation_type == 'Demographic':
        selected_columns = ["Age", "Gender", "Income Level", "Education Level", "Occupation", "Marital Status",
                            "Insurance Products Owned"]
    elif segmentation_type == 'Geographic':

        selected_columns = ["Country", "City", "Region"]
    else:

        selected_columns = ["Age", "Gender", "Income Level"]

    df_selected = df[selected_columns].copy()

    for col in df_selected.select_dtypes(include=['object']).columns:
        df_selected[col] = LabelEncoder().fit_transform(df_selected[col].astype(str))

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_selected)

    return df_scaled
