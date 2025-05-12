from volterra_gcn import (GraphVolterraVNN,
                          get_combinatorial_laplacian_np,
                          get_path_graph_laplacian_np)
import numpy as np
import networkx as nx
from sklearn.preprocessing import LabelEncoder
import torch
from typing import Dict
import os
from scipy.linalg import eigh
from torch import nn, optim
from tqdm import tqdm


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

    return connectivity_matrices, eeg_data, labels, order

def prepare_dataset(N_temporal, N_spatial):

    train_adj, train_X, train_labels, order = read_files("train")
    _, test_X, test_labels, _ = read_files("test", order=order)

    connectivity_matrix = np.mean(train_adj, axis=0)
    connectivity_matrix = (connectivity_matrix > 0.2)

    selected_nodes = select_nodes(connectivity_matrix, num_nodes=N_spatial_nodes)
    connectivity_matrix = connectivity_matrix[selected_nodes][:, selected_nodes]
    train_X = train_X[:, :, selected_nodes]
    test_X = test_X[:, :, selected_nodes]

    train_X = average_in_batches(train_X, batch_size=int(120/N_temporal_nodes))
    test_X = average_in_batches(test_X, batch_size=int(120 / N_temporal_nodes))

    train_X = train_X.reshape(train_X.shape[0], -1)
    test_X = test_X.reshape(test_X.shape[0], -1)

    return connectivity_matrix, train_X, train_labels, test_X, test_labels


if __name__ == "__main__":

    N_temporal_nodes = 120
    N_spatial_nodes = 69
    K_T1_user = 5          
    K_T2_user = 5     
    num_classes = 12     
    
    connectivity_matrix, train_X, train_labels, test_X, test_labels = prepare_dataset(N_temporal_nodes, N_spatial_nodes)

    L_G_np = get_combinatorial_laplacian_np(connectivity_matrix)
    _, U_G_np = eigh(L_G_np); sort_idx_G = eigh(L_G_np); sort_idx_G = np.argsort(_); U_G_np = U_G_np[:, sort_idx_G]

    L_P_T_np = get_path_graph_laplacian_np(N_temporal_nodes)
    _, U_P_T_np = eigh(L_P_T_np); sort_idx_P_T = np.argsort(_); U_P_T_np = U_P_T_np[:, sort_idx_P_T]

    model = GraphVolterraVNN(U_G_np, U_P_T_np, K_T1_user, K_T2_user, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_X = torch.from_numpy(train_X).float()
    test_X = torch.from_numpy(test_X).float()
    train_labels = torch.from_numpy(train_labels).long()
    test_labels = torch.from_numpy(test_labels).long()

    train_dataset = torch.utils.data.TensorDataset(train_X, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(test_X, test_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    print(f"Number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    for epoch in range(100):
        model.train()
        epoch_loss = 0.0
        epoch_correct_preds = 0
        epoch_total_preds = 0
        
        for s_batch, y_batch in tqdm(train_loader):
            optimizer.zero_grad()
            
            logits = model(s_batch)
            loss = criterion(logits, y_batch)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * s_batch.size(0)
            
            _, predicted_labels = torch.max(logits, 1)
            epoch_total_preds += y_batch.size(0)
            epoch_correct_preds += (predicted_labels == y_batch).sum().item()

        avg_epoch_loss = epoch_loss / epoch_total_preds
        avg_epoch_acc = epoch_correct_preds / epoch_total_preds
        
        print(f"Epoch {epoch+1}/{15} - "
              f"Loss: {avg_epoch_loss:.4f} - "
              f"Accuracy: {avg_epoch_acc:.4f} - ")
    
    train_acc = model.compute_accuracy(train_X, train_labels)
    test_acc = model.compute_accuracy(test_X, test_labels)

    print("Train Accuracy: ", train_acc)
    print("Test Accuracy: ", test_acc)