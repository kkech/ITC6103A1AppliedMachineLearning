import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

file_path = 'usCensusData/USCensus1990.data.txt'
data = pd.read_csv(file_path)
print("First few rows of the dataset:")
print(data.head())
print("\nDataFrame Info:")
data.info()
print("\nDescriptive Statistics:")
print(data.describe())
print("\nMissing Values in Each Column:")
print(data.isnull().sum())

features = data.drop(['caseid'], axis=1, errors='ignore')
# We limit the dataset to the first 1000 rows. Remove it if you want the full dataset
features = features[:1000]

### K-MEANS Starts ###

print("K-MEANS Starts")
plot_dir = 'k-means'
os.makedirs(plot_dir, exist_ok=True)
print("\nPerforming data standardization...")
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

def find_optimal_clusters(data, range_n_clusters):
    print("\nFinding the optimal number of clusters...")
    silhouette_avg_scores = []
    inertia_scores = []
    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        inertia = kmeans.inertia_
        silhouette_avg_scores.append(silhouette_avg)
        inertia_scores.append(inertia)
        print(f"Clusters: {n_clusters}, Silhouette Score: {silhouette_avg}")
    optimal_n_clusters = range_n_clusters[np.argmax(silhouette_avg_scores)]
    print(f"Optimal number of clusters: {optimal_n_clusters}")
    return optimal_n_clusters, silhouette_avg_scores, inertia_scores

range_n_clusters = list(range(2, 11))
optimal_n_clusters_full, silhouette_scores_full, inertia_scores_full  = find_optimal_clusters(scaled_features, range_n_clusters)

print("\nReducing data to 2 principal components for visualization...")
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_features)

optimal_n_clusters_pca, silhouette_scores_pca, inertia_scores_pca  = find_optimal_clusters(reduced_data, range_n_clusters)
print("Plotting Evaluation Metrics for K-Means...")
def plot_metrics(range_n_clusters, silhouette_scores, inertia_scores, title_suffix):
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range_n_clusters, silhouette_scores, marker='o', color='blue')
    plt.title(f'Silhouette Scores {title_suffix}')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')

    plt.subplot(1, 2, 2)
    plt.plot(range_n_clusters, inertia_scores, marker='o', color='red')
    plt.title(f'Inertia {title_suffix}')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/metrics_{title_suffix.replace(' ', '_').lower()}.png")
    plt.show()

plot_metrics(range_n_clusters, silhouette_scores_full, inertia_scores_full, "Full Feature Set")
plot_metrics(range_n_clusters, silhouette_scores_pca, inertia_scores_pca, "PCA-Reduced Data")

# Apply K-Means to the full feature set with the optimal number of clusters
print("\nApplying K-Means Clustering to the full feature set with the optimal number of clusters...")
kmeans_full = KMeans(n_clusters=optimal_n_clusters_full, random_state=42)
cluster_labels_full = kmeans_full.fit_predict(scaled_features)
centers_full = kmeans_full.cluster_centers_
distances_full = pairwise_distances(centers_full)
np.fill_diagonal(distances_full, np.inf)  # Replace zeros with 'inf' to ignore self-distances
outlier_cluster_full = np.argmax(np.min(distances_full, axis=1))
print(f"Outlier cluster (full feature set): {outlier_cluster_full}")

print("\nApplying K-Means Clustering to PCA-reduced data with the optimal number of clusters...")
kmeans_pca = KMeans(n_clusters=optimal_n_clusters_pca, random_state=42)
cluster_labels_pca = kmeans_pca.fit_predict(reduced_data)
centers_pca = kmeans_pca.cluster_centers_
distances_pca = pairwise_distances(centers_pca)
np.fill_diagonal(distances_pca, np.inf)
outlier_cluster_pca = np.argmax(np.min(distances_pca, axis=1))
print(f"Outlier cluster (PCA-reduced): {outlier_cluster_pca}")

def visualize_kmeans_clusters_with_outlier(data, labels, centers, outlier_cluster, title, filename):
    plt.figure(figsize=(10, 6))
    n_clusters = len(np.unique(labels))
    colors = plt.cm.tab10(np.arange(n_clusters) / n_clusters)
    for i, color in zip(range(n_clusters), colors):
        if i == outlier_cluster:
            plt.scatter(data[labels == i, 0], data[labels == i, 1], s=100, color='black', label=f'Outlier Cluster {i}', marker='P', edgecolors='w', linewidths=0.5)
        else:
            plt.scatter(data[labels == i, 0], data[labels == i, 1], s=50, color=color, label=f'Cluster {i}', alpha=0.7)
    plt.scatter(centers[:, 0], centers[:, 1], s=250, marker='*', c='red', label='Centroids')
    plt.title(title)
    plt.xlabel('Feature 1' if 'All Features' in title else 'PCA Component 1')
    plt.ylabel('Feature 2' if 'All Features' in title else 'PCA Component 2')
    plt.legend(loc='best')
    plt.savefig(f"{plot_dir}/{filename}")
    plt.show()

visualize_kmeans_clusters_with_outlier(scaled_features, cluster_labels_full, centers_full, outlier_cluster_full, 'K-Means Clustering with All Features', 'kmeans_with_outlier_full_features.png')
visualize_kmeans_clusters_with_outlier(reduced_data, cluster_labels_pca, centers_pca, outlier_cluster_pca, 'K-Means Clustering in PCA-Reduced Space', 'kmeans_with_outlier_pca_reduced.png')

print("K-MEANS End")

### K-MEANS Ends ###

### AHC Starts ###

print("Agglomerative Hierarchical Clustering (AHC) Starts")
plot_dir = 'ahc'
os.makedirs(plot_dir, exist_ok=True)
print("\nPerforming data standardization...")
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

def find_optimal_clusters(data, range_n_clusters):
    print("\nFinding the optimal number of clusters...")
    silhouette_avg_scores = []
    for n_clusters in range_n_clusters:
        ahc = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = ahc.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_avg_scores.append(silhouette_avg)
        print(f"Clusters: {n_clusters}, Silhouette Score: {silhouette_avg}")
    optimal_n_clusters = range_n_clusters[np.argmax(silhouette_avg_scores)]
    print(f"Optimal number of clusters: {optimal_n_clusters}")
    return optimal_n_clusters, silhouette_avg_scores

range_n_clusters = list(range(2, 11))
optimal_n_clusters_full, silhouette_scores_full = find_optimal_clusters(scaled_features, range_n_clusters)

print("\nReducing data to 2 principal components for visualization...")
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_features)
optimal_n_clusters_pca, silhouette_scores_pca = find_optimal_clusters(reduced_data, range_n_clusters)
print("Plotting Evaluation Metrics for AHC...")
def plot_metrics(range_n_clusters, silhouette_scores, title_suffix):
    plt.figure(figsize=(10, 5))
    plt.plot(range_n_clusters, silhouette_scores, marker='o', color='blue')
    plt.title(f'Silhouette Scores {title_suffix}')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/silhouette_{title_suffix.replace(' ', '_').lower()}.png")
    plt.close()

plot_metrics(range_n_clusters, silhouette_scores_full, "Full Feature Set")
plot_metrics(range_n_clusters, silhouette_scores_pca, "PCA-Reduced Data")

print("\nVisualizing dendrogram for full feature set...")
plt.figure(figsize=(10, 7))
Z = linkage(scaled_features, 'ward')
dendrogram(Z)
plt.title('Dendrogram for AHC - Full Feature Set')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.savefig(f"{plot_dir}/dendrogram_full_feature_set.png")
plt.close()

print("\nVisualizing dendrogram for PCA-reduced data...")
plt.figure(figsize=(10, 7))
Z_reduced = linkage(reduced_data, 'ward')
dendrogram(Z_reduced)
plt.title('Dendrogram for AHC - PCA-Reduced Data')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.savefig(f"{plot_dir}/dendrogram_pca_reduced.png")
plt.close()

ahc_full = AgglomerativeClustering(n_clusters=optimal_n_clusters_full)
labels_full = ahc_full.fit_predict(scaled_features)
ahc_pca = AgglomerativeClustering(n_clusters=optimal_n_clusters_pca)
labels_pca = ahc_pca.fit_predict(reduced_data)


def analyze_cluster_silhouette_scores(data, cluster_labels):
    silhouette_vals = silhouette_samples(data, cluster_labels)
    cluster_silhouette_vals = {}
    for i in range(len(cluster_labels)):
        cluster_label = cluster_labels[i]
        if cluster_label not in cluster_silhouette_vals:
            cluster_silhouette_vals[cluster_label] = []
        cluster_silhouette_vals[cluster_label].append(silhouette_vals[i])
    
    avg_silhouette_scores = {cluster: np.mean(scores) for cluster, scores in cluster_silhouette_vals.items()}
    outlier_cluster = min(avg_silhouette_scores, key=avg_silhouette_scores.get)
    
    return avg_silhouette_scores, outlier_cluster

avg_silhouette_scores_full, outlier_cluster_full = analyze_cluster_silhouette_scores(scaled_features, labels_full)
avg_silhouette_scores_pca, outlier_cluster_pca = analyze_cluster_silhouette_scores(reduced_data, labels_pca)
print(f"Full feature set - Average silhouette scores per cluster: {avg_silhouette_scores_full}")
print(f"Outlier cluster (full feature set): {outlier_cluster_full}")
print(f"PCA-reduced data - Average silhouette scores per cluster: {avg_silhouette_scores_pca}")
print(f"Outlier cluster (PCA-reduced): {outlier_cluster_pca}")

def visualize_clusters_with_outliers(data, cluster_labels, outlier_cluster_label, title, filename):
    plt.figure(figsize=(10, 7))
    unique_labels = np.unique(cluster_labels)
    for label in unique_labels:
        cluster_data = data[cluster_labels == label]
        if label == outlier_cluster_label:
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=100, label=f'Outlier Cluster {label}', edgecolors='black', linewidths=1.5, alpha=0.85, marker='*')
        else:
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], s=50, label=f'Cluster {label}', alpha=0.7)
    plt.title(title)
    plt.xlabel('Feature 1' if 'Full' in title else 'PCA Component 1')
    plt.ylabel('Feature 2' if 'Full' in title else 'PCA Component 2')
    plt.legend()
    plt.savefig(f"{plot_dir}/{filename}")
    plt.show()

visualize_clusters_with_outliers(scaled_features, labels_full, outlier_cluster_full, 'AHC Clusters with Outlier - Full Features', 'ahc_clusters_outlier_full.png')
visualize_clusters_with_outliers(reduced_data, labels_pca, outlier_cluster_pca, 'AHC Clusters with Outlier - PCA Reduced', 'ahc_clusters_outlier_pca.png')

print("AHC Analysis Ends")