import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import griddata


def read_data(filename="sensitivity.csv"):
    df = pd.read_csv(filename)
    return df.values


def get_matrixes(data_values, max_cells=72000, max_comps=24):
    mat_var = [[0 for _ in range(max_cells // max_comps)] for _ in range(max_comps)]
    mat_data_reduction = [[0 for _ in range(max_cells // max_comps)] for _ in range(max_comps)]
    mat_heat_score = [[0 for _ in range(max_cells // max_comps)] for _ in range(max_comps)]
    """
    comps_vect, clusters_vect = np.array([i for i in range(1, max_comps + 1)]), np.array([i for i in range(1, max_cells // max_comps + 1)]), 
    var_vect, data_reduction_vect, heat_score_vect = np.array([]), np.array([]), np.array([])
    """
    for row in data_values:
        n_comps, n_clusters = int(row[0]), int(row[1])

        explained_variance_tot = row[-1]
        dataset_reduction = 1 - (n_comps * n_clusters) / max_cells
        heat_score = explained_variance_tot * dataset_reduction
        """
        var_vect = np.append(var_vect, explained_variance_tot)
        data_reduction_vect = np.append(data_reduction_vect, dataset_reduction)
        heat_score_vect = np.append(heat_score_vect, heat_score)
        """
        mat_var[n_comps - 1][n_clusters - 1] = explained_variance_tot
        mat_data_reduction[n_comps - 1][n_clusters - 1] = dataset_reduction
        mat_heat_score[n_comps - 1][n_clusters - 1] = heat_score

    """
    comps_axis = np.linspace(comps_vect.min(), comps_vect.max(), max_cells // max_comps)
    clusters_axis = np.linspace(clusters_vect.min(), clusters_vect.max(), max_cells // max_comps)
    mat_var = griddata((n_comps, n_clusters), var_vect, (comps_axis[None,:], clusters_axis[:,None]), method='cubic')
    mat_data_reduction = griddata((n_comps, n_clusters), data_reduction_vect, (comps_axis[None,:], clusters_axis[:,None]), method='cubic')
    mat_heat_score = griddata((n_comps, n_clusters), heat_score_vect, (comps_axis[None,:], clusters_axis[:,None]), method='cubic')
    """
    return mat_var, mat_data_reduction, mat_heat_score


def heatmap(mat, fig_name, colormap):
    extent = [1, len(mat[0]), len(mat), 1]
    plt.imshow(mat, cmap=colormap, interpolation='nearest', aspect='auto', extent=extent)
    plt.xlabel('Number of clusters')
    plt.ylabel('Number of principal components')
    plt.colorbar()
    plt.savefig(fig_name)
    plt.close()


if __name__ == "__main__":
    data_values = read_data()
    mat_var, mat_data_reduction, mat_heat_score = get_matrixes(data_values)
    cmap1 = "Set1"
    heatmap(mat_var, "explained_var_sensitivity_qualitative.png", cmap1)
    heatmap(mat_data_reduction, "data_reduction_sensitivity_qualitative.png", cmap1)
    heatmap(mat_heat_score, "heat_score_sensitivity_qualitative.png", cmap1)

    cmap2 = "nipy_spectral"
    heatmap(mat_var, "explained_var_sensitivity.png", cmap2)
    heatmap(mat_data_reduction, "data_reduction_sensitivity.png", cmap2)
    heatmap(mat_heat_score, "heat_score_sensitivity.png", cmap2)
