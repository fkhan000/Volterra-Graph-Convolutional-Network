import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.linalg import eigh
import time

# --- Helper Functions (can remain in NumPy, results converted to tensors later) ---
def get_combinatorial_laplacian_np(adj_matrix):
    N = adj_matrix.shape[0]
    deg_matrix_diag = np.sum(adj_matrix.astype(float), axis=1)
    deg_matrix = np.diag(deg_matrix_diag)
    return deg_matrix - adj_matrix.astype(float)

def get_path_graph_laplacian_np(T):
    if T == 1: return np.array([[0.0]])
    adj_P_T = np.zeros((T, T))
    for i in range(T - 1):
        adj_P_T[i, i + 1] = 1.0
        adj_P_T[i + 1, i] = 1.0
    return get_combinatorial_laplacian_np(adj_P_T)

# --- PyTorch GraphVolterraLayer ---
class GraphVolterraLayer(nn.Module):
    def __init__(self, U_G_np, U_P_T_np, K_T1, K_T2):
        super().__init__()

        self.N_spatial = U_G_np.shape[0]
        self.N_temporal = U_P_T_np.shape[0]
        self.N_effective = self.N_spatial * self.N_temporal

        self.K_T1 = K_T1 # Num basis functions for h_hat_1's temporal part
        self.K_T2 = K_T2 # Num basis functions for H_hat_2_T

        if self.K_T1 > self.N_temporal:
            raise ValueError(f"K_T1 ({self.K_T1}) cannot exceed N_temporal ({self.N_temporal})")
        if self.K_T2 > self.N_temporal:
            raise ValueError(f"K_T2 ({self.K_T2}) cannot exceed N_temporal ({self.N_temporal})")

        # Register GFT matrices as non-trainable buffers
        # Convert NumPy arrays to PyTorch tensors
        self.register_buffer('U_G', torch.from_numpy(U_G_np.astype(np.float32)))
        self.register_buffer('U_P_T', torch.from_numpy(U_P_T_np.astype(np.float32)))
        
        U_ST_tensor = torch.kron(self.U_G, self.U_P_T)
        self.register_buffer('U_ST', U_ST_tensor)
        
        U_P_T_basis1_tensor = self.U_P_T[:, :self.K_T1]
        self.register_buffer('U_P_T_basis1', U_P_T_basis1_tensor) # N_T x K_T1
        
        U_P_T_basis2_tensor = self.U_P_T[:, :self.K_T2]
        self.register_buffer('U_P_T_basis2', U_P_T_basis2_tensor) # N_T x K_T2

        # Learnable Volterra Parameters
        # Initialize with small random values (can be improved with Kaiming/Xavier init)
        self.h0 = nn.Parameter(torch.randn(self.N_effective) * 0.01)
        self.h_hat_1_coeffs = nn.Parameter(torch.randn(self.N_spatial, self.K_T1) * 0.01) # N_S x K_T1
        self.H_hat_2_S = nn.Parameter(torch.randn(self.N_spatial, self.N_spatial) * 0.01) # N_S x N_S
        self.H_hat_2_T_coeffs = nn.Parameter(torch.randn(self.K_T2, self.K_T2) * 0.01) # K_T2 x K_T2
        
        # Ensure symmetry for H_hat_2_S and H_hat_2_T_coeffs if desired during/after training
        # For initialization, one could do:
        # H_S_init = torch.randn(self.N_spatial, self.N_spatial) * 0.01
        # self.H_hat_2_S = nn.Parameter((H_S_init + H_S_init.T) / 2)
        # H_T_c_init = torch.randn(self.K_T2, self.K_T2) * 0.01
        # self.H_hat_2_T_coeffs = nn.Parameter((H_T_c_init + H_T_c_init.T) / 2)


    def _gft(self, signal_batch, U):
        # signal_batch: (batch_size, N_effective)
        # U: (N_effective, N_effective)
        # Output: (batch_size, N_effective)
        return torch.matmul(signal_batch, U) # Using U for GFT (U.T s), so U s for iGFT if U is orthonormal
                                             # If U columns are eigenvectors, U.T @ s for GFT
                                             # s_hat = U.T @ s_col --> s_hat_row = s_row @ U
                                             # Let's assume signal_batch is (batch, N_eff)
                                             # GFT: s_hat = signal_batch @ U (if U's columns are eigenvectors u_i, then U_ij = u_j[i])
                                             # This is equivalent to (U.T @ s_col_vector).T
                                             # If U is (N,N) matrix of eigenvectors as columns, U.T @ x
                                             # For row vectors x_row @ U
        # For U.T @ signal (where signal is column vector)
        # (batch_size, N_effective) @ (N_effective, N_effective) -> (batch_size, N_effective)
        return torch.matmul(signal_batch.unsqueeze(1), self.U_ST.T).squeeze(1)


    def _igft(self, spectral_signal_batch, U):
        # spectral_signal_batch: (batch_size, N_effective)
        # U: (N_effective, N_effective)
        # Output: (batch_size, N_effective)
        # Inverse GFT: signal = U @ spectral_signal_col_vector
        # For row vectors: spectral_signal_row @ U.T
        return torch.matmul(spectral_signal_batch.unsqueeze(1), self.U_ST).squeeze(1)


    def _reconstruct_full_h_hat_1(self):
        # h_hat_1_coeffs: (N_S, K_T1)
        # U_P_T_basis1: (N_T, K_T1)
        # h_hat_1_matrix_form = h_hat_1_coeffs @ U_P_T_basis1.T  -> (N_S, N_T)
        h_hat_1_matrix_form = torch.matmul(self.h_hat_1_coeffs, self.U_P_T_basis1.T)
        # Flatten (order='C'): spatial index varies slower
        return h_hat_1_matrix_form.contiguous().view(-1) # Shape: (N_effective)

    def _reconstruct_full_H_hat_2_T(self):
        # H_hat_2_T_coeffs: (K_T2, K_T2)
        # U_P_T_basis2: (N_T, K_T2)
        # H_hat_2_T = U_P_T_basis2 @ H_hat_2_T_coeffs @ U_P_T_basis2.T
        temp = torch.matmul(self.H_hat_2_T_coeffs, self.U_P_T_basis2.T)
        return torch.matmul(self.U_P_T_basis2, temp) # Shape: (N_T, N_T)

    def _get_effective_H_hat_2(self):
        full_H_hat_2_T = self._reconstruct_full_H_hat_2_T() # N_T x N_T
        # H_hat_2_S: N_S x N_S
        return torch.kron(self.H_hat_2_S, full_H_hat_2_T) # N_eff x N_eff

    def forward(self, s_effective_batch):
        # s_effective_batch: (batch_size, N_effective)
        batch_size = s_effective_batch.shape[0]

        # Zeroth-order term
        # h0: (N_effective), expand for batch: (batch_size, N_effective)
        z0 = self.h0.unsqueeze(0).expand(batch_size, -1)

        # First-order term
        # s_hat_batch: (batch_size, N_effective)
        s_hat_batch = self._gft(s_effective_batch, self.U_ST) 
        
        full_h_hat_1 = self._reconstruct_full_h_hat_1() # (N_effective)
        # full_h_hat_1_batch: (batch_size, N_effective)
        full_h_hat_1_batch = full_h_hat_1.unsqueeze(0).expand(batch_size, -1)
        
        z1_hat_batch = full_h_hat_1_batch * s_hat_batch # Element-wise
        z1 = self._igft(z1_hat_batch, self.U_ST) # (batch_size, N_effective)

        # Second-order term
        # This is the most computationally intensive part for batch processing
        H2_effective = self._get_effective_H_hat_2() # (N_eff, N_eff) - computed once

        # s_hat_batch: (B, N_eff)
        # We need (s_hat @ s_hat.T) for each sample, then multiply by H2_effective
        # s_hat_outer_prods_batch: (B, N_eff, N_eff)
        # s_hat_batch.unsqueeze(2) -> (B, N_eff, 1)
        # s_hat_batch.unsqueeze(1) -> (B, 1, N_eff)
        s_hat_outer_prods_batch = torch.bmm(s_hat_batch.unsqueeze(2), s_hat_batch.unsqueeze(1))
        
        # Y_hat_prod_batch: (B, N_eff, N_eff)
        # H2_effective is (N_eff, N_eff). Need to broadcast multiplication.
        # H2_eff_b = H2_effective.unsqueeze(0).expand(batch_size, -1, -1) -> (B, N_eff, N_eff)
        # Y_hat_prod_batch = H2_eff_b * s_hat_outer_prods_batch
        # More efficiently using einsum for (H_eff * (sh @ sh.T)) element-wise then sum
        # Or, reshape H2_effective and s_hat_outer_prods_batch for element-wise product
        # Y_hat_prod_batch = H2_effective * s_hat_outer_prods_batch # Element-wise if H2_eff is (B,N,N)

        # Let's do it sample by sample for clarity, can be optimized with torch.einsum or careful matmuls
        z2_list = []
        for i in range(batch_size):
            s_h_i = s_hat_batch[i, :].unsqueeze(1) # (N_eff, 1)
            s_h_outer_i = torch.matmul(s_h_i, s_h_i.T) # (N_eff, N_eff)
            Y_hat_prod_i = H2_effective * s_h_outer_i # Element-wise
            
            # Y_prod_i = U_ST @ Y_hat_prod_i @ U_ST.T
            Y_prod_i = torch.matmul(torch.matmul(self.U_ST, Y_hat_prod_i), self.U_ST.T)
            z2_list.append(torch.diag(Y_prod_i)) # (N_eff)
        
        z2 = torch.stack(z2_list) # (batch_size, N_effective)

        output_signal_batch = z0 + z1 + z2 # (batch_size, N_effective)
        return output_signal_batch

class GraphVolterraVNN(nn.Module):
    def __init__(self, U_G_np, U_P_T_np, K_T1, K_T2, num_classes):
        super().__init__()
        self.volterra_layer = GraphVolterraLayer(U_G_np, U_P_T_np, K_T1, K_T2)
        self.N_effective = self.volterra_layer.N_effective
        
        # Final classification layer
        self.classifier_head = nn.Linear(self.N_effective, num_classes)

    def forward(self, s_effective_batch):
        # s_effective_batch: (batch_size, N_effective)
        volterra_features = self.volterra_layer(s_effective_batch) # (batch_size, N_effective)
        logits = self.classifier_head(volterra_features) # (batch_size, num_classes)
        return logits
    
    def predict(self, X):
        class_probabilities = self.forward(X)
        return torch.argmax(class_probabilities, dim=1)
    
    def compute_accuracy(self, X, labels):
        predicted_labels = self.predict(X)
        correct = (predicted_labels == labels).sum().item()
        total = labels.size(0)
        return correct / total

# --- Example Usage and Training Loop ---
if __name__ == '__main__':
    # --- Configuration ---
    N_spatial_nodes = 5    
    N_temporal_points = 8 
    K_T1_user = 3          
    K_T2_user = 3          
    num_classes_example = 4 
    
    batch_size_example = 16
    num_epochs_example = 20
    learning_rate_example = 1e-3
    
    # --- Generate Dummy GFT Matrices (replace with your actual U_G, U_P_T) ---
    np.random.seed(42)
    # Dummy Adjacency for Spatial Graph
    adj_G_np = (np.random.rand(N_spatial_nodes, N_spatial_nodes) < 0.5).astype(float)
    np.fill_diagonal(adj_G_np, 0); adj_G_np = (adj_G_np + adj_G_np.T)/2; adj_G_np[adj_G_np > 0.1] = 1
    L_G_np = get_combinatorial_laplacian_np(adj_G_np)
    _, U_G_np = eigh(L_G_np); sort_idx_G = np.argsort(_); U_G_np = U_G_np[:, sort_idx_G]
    
    # Dummy Path Graph for Temporal
    L_P_T_np = get_path_graph_laplacian_np(N_temporal_points)
    _, U_P_T_np = eigh(L_P_T_np); sort_idx_P_T = np.argsort(_); U_P_T_np = U_P_T_np[:, sort_idx_P_T]

    # --- Initialize Model, Loss, Optimizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GraphVolterraVNN(U_G_np, U_P_T_np, K_T1_user, K_T2_user, num_classes_example).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_example)

    # --- Generate Dummy Training Data (replace with your actual data loader) ---
    N_effective_nodes = N_spatial_nodes * N_temporal_points
    num_train_samples = 100
    
    # S_train_effective_np should be (num_train_samples, N_effective)
    # In your case, from (num_samples, N_T, N_S) -> transpose to (N_S, N_T) -> flatten 'C'
    # Example: raw_data (num_samples, N_T, N_S)
    # S_train_effective_np = np.array([s.T.flatten(order='C') for s in raw_data_train])
    S_train_effective_np = np.random.randn(num_train_samples, N_effective_nodes).astype(np.float32)
    # Y_train_np should be (num_train_samples,) with class indices
    Y_train_np = np.random.randint(0, num_classes_example, size=num_train_samples)

    S_train_torch = torch.from_numpy(S_train_effective_np).to(device)
    Y_train_torch = torch.from_numpy(Y_train_np).long().to(device) # CrossEntropyLoss expects long type for labels

    # Create a simple DataLoader
    train_dataset = torch.utils.data.TensorDataset(S_train_torch, Y_train_torch)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_example, shuffle=True)

    print(f"Model initialized. N_effective: {model.N_effective}")
    print(f"Number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(num_epochs_example):
        model.train() # Set model to training mode
        epoch_loss = 0.0
        epoch_correct_preds = 0
        epoch_total_preds = 0
        
        start_epoch_time = time.time()
        for s_batch, y_batch in train_loader:
            optimizer.zero_grad() # Zero gradients for every batch
            
            logits = model(s_batch) # Forward pass
            loss = criterion(logits, y_batch) # Compute loss
            
            loss.backward() # Backward pass: compute gradient of the loss w.r.t. parameters
            optimizer.step() # Update parameters
            
            epoch_loss += loss.item() * s_batch.size(0)
            
            _, predicted_labels = torch.max(logits, 1)
            epoch_total_preds += y_batch.size(0)
            epoch_correct_preds += (predicted_labels == y_batch).sum().item()

        avg_epoch_loss = epoch_loss / epoch_total_preds
        avg_epoch_acc = epoch_correct_preds / epoch_total_preds
        end_epoch_time = time.time()
        
        print(f"Epoch {epoch+1}/{num_epochs_example} - "
              f"Loss: {avg_epoch_loss:.4f} - "
              f"Accuracy: {avg_epoch_acc:.4f} - "
              f"Time: {end_epoch_time - start_epoch_time:.2f}s")

    print("Training complete.")

    # --- Example: Getting features from the Volterra layer (after training) ---
    # model.eval()
    # with torch.no_grad():
    #     dummy_input = torch.randn(5, N_effective_nodes).to(device) # 5 samples
    #     volterra_features = model.volterra_layer(dummy_input)
    #     print(f"Shape of extracted Volterra features: {volterra_features.shape}") # (5, N_effective)
    #     final_predictions = model(dummy_input)
    #     print(f"Shape of final predictions: {final_predictions.shape}") # (5, num_classes)

