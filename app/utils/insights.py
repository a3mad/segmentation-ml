import numpy as np


def generate_insights(data, labels):
    n_clusters = len(set(labels))
    insights = []
    for c in range(n_clusters):
        cluster_indices = np.where(labels == c)
        cluster_data = data[cluster_indices]

        insights.append({
            'cluster_id': c,
            'description': f"Cluster {c} might represent a group with unique characteristics.",
            'recommendation': "Consider targeted marketing efforts for this segment."
        })
    return insights
