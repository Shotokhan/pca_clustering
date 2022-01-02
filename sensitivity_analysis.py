import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import csv


def get_values(filename="data.csv"):
    df = pd.read_csv(filename)
    return df.values


def normalize_data(data_values):
    # this is z-score: mean=0 and variance=1
    norm = StandardScaler().fit_transform(data_values)
    return norm


def do_pca(norm_data, n_components):
    assert n_components > 0 and n_components <= len(norm_data[0])
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(norm_data)
    explained_variance = sum(pca.explained_variance_ratio_)
    return principalComponents, min(explained_variance, 1)


def pca_stats(p_comps):
    mean_p_comps = sum(p_comps) / (len(p_comps) - 1)
    dev_pca = sum([sum((row - mean_p_comps) ** 2) for row in p_comps])
    return mean_p_comps, dev_pca


def do_clustering(p_comps, n_clusters, mean_p_comps, dev_pca):
    assert n_clusters > 0 and n_clusters <= len(p_comps)
    hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    y_hc = hc.fit_predict(p_comps)
    clusters = {i: [] for i in set(y_hc)}
    for i in range(len(p_comps)):
        clusters[y_hc[i]].append(p_comps[i])
    centroids = [sum(clusters[i]) / (len(clusters[i])) for i in range(len(clusters))]
    inter_cluster_deviance = sum([len(clusters[k]) * sum((centroids[k] - mean_p_comps) ** 2) for k in range(len(clusters))])
    """
    intra_cluster_deviance = sum([sum([sum((clusters[k][i] - centroids[k]) ** 2) for i in range(len(clusters[k]))]) for k in range(len(clusters))])
    try:
        assert abs((intra_cluster_deviance + inter_cluster_deviance) - dev_pca) < 1e-3
    except AssertionError:
        print(f"Error: {intra_cluster_deviance = }, {inter_cluster_deviance = }, {dev_pca = }")
    """
    explained_variance = inter_cluster_deviance / dev_pca
    return centroids, min(explained_variance, 1)


if __name__ == "__main__":
    data_values = get_values()
    norm_data = normalize_data(data_values)
    rows, fields = [], ["n_components", "n_clusters", "explained_variance_pca", "explained_variance_clustering", "explained_variance_tot"]
    for i in range(1, len(norm_data[0]) + 1):
        p_comps, explained_variance_pca = do_pca(norm_data, i)
        mean_p_comps, dev_pca = pca_stats(p_comps)
        for n_clusters in range(1, len(p_comps) + 1):
            _, explained_variance_clustering = do_clustering(p_comps, n_clusters, mean_p_comps, dev_pca)
            explained_variance = explained_variance_pca * explained_variance_clustering
            print(f"Explained variance {explained_variance} with {i} components and {n_clusters} clusters; {explained_variance_pca = }, {explained_variance_clustering = }")
            rows.append([i, n_clusters, explained_variance_pca, explained_variance_clustering, explained_variance])
    with open("sensitivity.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for row in rows:
            writer.writerow(row)

