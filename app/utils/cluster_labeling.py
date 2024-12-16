def generate_cluster_labels(cluster_data, cluster_labels, metadata):
    """
    Generate descriptive labels for clusters based on selected columns.
    :param cluster_data: DataFrame containing the original data
    :param cluster_labels: Cluster labels from the clustering algorithm
    :param metadata: Information about required and additional columns
    :return: A dictionary mapping cluster IDs to descriptive labels
    """
    cluster_data = cluster_data.reset_index(drop=True)
    cluster_data['Cluster'] = cluster_labels
    cluster_descriptions = {}

    # Retrieve encoders from metadata
    encoders = metadata.get('encoders', {})

    # Identify columns already handled as categorical
    categorical_columns = list(metadata['required_columns'].keys())
    additional_columns = metadata.get('additional_columns', [])

    # Calculate global means for numerical variables
    global_means = cluster_data.mean()

    # Exclude columns that are already handled as categorical
    numeric_columns = [
        col for col in cluster_data.select_dtypes(include='number').columns
        if col not in categorical_columns
    ]

    for cluster_id in sorted(cluster_data['Cluster'].unique()):
        cluster_subset = cluster_data[cluster_data['Cluster'] == cluster_id]
        label_parts = []

        # Handle Categorical Variables
        for col in categorical_columns:
            if col in cluster_subset:
                if col in encoders:
                    encoded_values = cluster_subset[col].mode()
                    if not encoded_values.empty:
                        original_value = encoders[col].inverse_transform([encoded_values[0]])[0]
                        label_parts.append(original_value)
                else:
                    #use raw value
                    common_value = cluster_subset[col].mode()[0]
                    label_parts.append(f"{col} around: "+str(common_value))

        # Include Additional Columns (if any)
        for col in additional_columns:
            if col in cluster_subset:
                common_value = cluster_subset[col].mode()[0]
                label_parts.append(f"{col} around: "+str(common_value))

        # Combine parts into a single label
        cluster_descriptions[cluster_id] = ", ".join(label_parts)

    return cluster_descriptions
