
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def perform_clustering(data):
    best_score = -1
    best_k = 2

    # Determine the optimal number of clusters
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        score = silhouette_score(data, labels)
        if score > best_score:
            best_score = score
            best_k = k

    # Final clustering with the best number of clusters
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    labels = kmeans.fit_predict(data)

    return labels, {
        'n_clusters': best_k,
        'centers': kmeans.cluster_centers_.tolist(),
        'silhouette_score': best_score
    }

