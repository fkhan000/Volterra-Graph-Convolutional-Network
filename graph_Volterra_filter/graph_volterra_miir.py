from graph_Volterra import (GraphVolterraModelTemporalBasis,
                            get_combinatorial_laplacian,
                            get_path_graph_laplacian,
                            gft,
                            igft)
import numpy as np
import os
from typing import Dict
from scipy.linalg import eigh
import networkx as nx

def read_files(split: str, order: Dict[str, int] = None):
    eeg_data = np.load(os.path.join("..", "data", "OpenMIIR", f"{split}_eeg_data.npy"))

    eeg_data = np.reshape(eeg_data, (-1, 120, 69))
    connectivity_matrices = np.load(os.path.join("..", "data", "OpenMIIR", f"{split}_trial_graphs.npy"))
    labels = []
    if order is None:
        order = dict()

    with open(os.path.join("..", "data", "OpenMIIR", f"{split}_labels.csv"), "r") as f:
        f.readline()
        for line in f:
            if line.strip() not in order:
                order[line.strip()] = len(order)
            labels.append(order[line.strip()])
    
    labels = np.array(labels)
    one_hot_labels = np.eye(len(order))[labels]

    return connectivity_matrices, eeg_data, one_hot_labels, order
        

def get_accuracy(class_probabilities, labels):
    predicted_labels = np.argmax(class_probabilities, axis = 1)
    true_labels = np.argmax(labels, axis=1)
    accuracy = np.mean(predicted_labels == true_labels)
    return accuracy

def select_nodes(adj_matrix, num_nodes=10):
    G = nx.from_numpy_array(adj_matrix)
    centrality_measure = nx.degree_centrality(G)
    selected_nodes = sorted(list(centrality_measure.keys()),
           key = centrality_measure.get,
           reverse=True
           )[:num_nodes]
    return np.array(selected_nodes)

def average_in_batches(x, batch_size):
    n_batches = x.shape[1] // batch_size
    x = x[:, :n_batches * batch_size, :]
    x = x.reshape(x.shape[0], n_batches, batch_size, x.shape[2])
    return np.mean(x, axis=2)


if __name__ == "__main__":
    
    train_adj, train_X, train_labels, order = read_files("train")
    _, test_X, test_labels, _ = read_files("test", order=order)
    N_temporal_nodes = 15
    N_spatial_nodes = 15

    connectivity_matrix = np.mean(train_adj, axis=0)
    connectivity_matrix = (connectivity_matrix > 0.2)

    selected_nodes = select_nodes(connectivity_matrix, num_nodes=N_spatial_nodes)
    connectivity_matrix = connectivity_matrix[selected_nodes][:, selected_nodes]
    train_X = train_X[:, :, selected_nodes]
    test_X = test_X[:, :, selected_nodes]

    train_X = average_in_batches(train_X, batch_size=int(120/N_temporal_nodes))
    test_X = average_in_batches(test_X, batch_size=int(120 / N_temporal_nodes))


    print("Train_X Shape: ", train_X.shape)
    print("Test_X Shape: ", test_X.shape)
    print("Adjacency Matrix Shape: ", connectivity_matrix.shape)

    L_G = get_combinatorial_laplacian(connectivity_matrix)
    eigvals_G, U_G = eigh(L_G); sort_idx_G = np.argsort(eigvals_G); U_G = U_G[:, sort_idx_G]

    L_P_T = get_path_graph_laplacian(N_temporal_nodes)
    eigvals_P_T, U_P_T = eigh(L_P_T); sort_idx_P_T = np.argsort(eigvals_P_T); U_P_T = U_P_T[:, sort_idx_P_T]

    volterra_model = GraphVolterraModelTemporalBasis(U_G, U_P_T, 10, 10, num_classes=12)
    volterra_model.fit(train_X.reshape(train_X.shape[0], -1, order='C'), train_labels, l2_reg=1e-4)

    class_probabilities = np.array([
        volterra_model.forward(train_X[i].flatten(order='C'), apply_linear=True) 
        for i in range(train_X.shape[0])
    ])
    train_accuracy = get_accuracy(class_probabilities, train_labels)

    class_probabilities = np.array([
        volterra_model.forward(test_X[i].flatten(order='C'), apply_linear=True) 
        for i in range(test_X.shape[0])
    ])
    test_accuracy = get_accuracy(class_probabilities, test_labels)

    print("Train Accuracy: ", train_accuracy)
    print("Test Accuracy: ", test_accuracy)

