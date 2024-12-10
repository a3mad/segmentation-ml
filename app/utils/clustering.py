from sklearn.cluster import KMeans
import numpy as np


def perform_clustering(data):
    from sklearn.metrics import silhouette_score
    best_score = -1
    best_k = 2
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        if score > best_score:
            best_score = score
            best_k = k

    final_kmeans = KMeans(n_clusters=best_k, random_state=42)
    final_labels = final_kmeans.fit_predict(data)

    cluster_info = {
        'n_clusters': best_k,
        'centers': final_kmeans.cluster_centers_.tolist(),
        'silhouette_score': best_score
    }

    return final_labels, cluster_info
