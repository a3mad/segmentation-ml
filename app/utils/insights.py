import numpy as np


def generate_insights(original_df, cluster_labels):
    # Reset index to align with cluster labels
    original_df = original_df.reset_index(drop=True)

    # Add cluster labels to the DataFrame
    original_df['Cluster'] = cluster_labels

    insights = []
    for cluster_id in sorted(original_df['Cluster'].unique()):
        cluster_data = original_df[original_df['Cluster'] == cluster_id]
        insights.append({
            'cluster_id': cluster_id,
            'description': f"Cluster {cluster_id} has {len(cluster_data)} customers.",
            'stats': cluster_data.describe(include='all').to_dict()
        })

    return insights

