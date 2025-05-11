import numpy as np
import scipy
from scipy.linalg import eigh
# from scipy.optimize import least_squares # Not used in this ALS version with direct lstsq
import time
<<<<<<< HEAD
from tqdm import tqdm
=======
>>>>>>> 394c7155ace262f89db83c8bdec54dae595b5c33

# Helper function to compute Graph Laplacian (combinatorial)
def get_combinatorial_laplacian(adj_matrix):
    N = adj_matrix.shape[0]
    deg_matrix_diag = np.sum(adj_matrix.astype(float), axis=1)
    deg_matrix = np.diag(deg_matrix_diag)
    return deg_matrix - adj_matrix.astype(float)

# Helper function to compute Path Graph Laplacian
def get_path_graph_laplacian(T):
    if T == 1: return np.array([[0.0]])
    adj_P_T = np.zeros((T, T))
    for i in range(T - 1):
        adj_P_T[i, i + 1] = 1.0
        adj_P_T[i + 1, i] = 1.0
    return get_combinatorial_laplacian(adj_P_T)

# Graph Fourier Transform and its inverse
def gft(signal, U): return U.T @ signal
def igft(spectral_signal, U): return U @ spectral_signal

class GraphVolterraModelTemporalBasis:
    """
    Spatio-Temporal Graph Volterra model with:
    - Separable 2nd order kernels: H_hat_2 = H_hat_2_S (x) H_hat_2_T
    - Temporal components (h_hat_1_T, H_hat_2_T) parameterized by basis functions
      (first K columns of U_P_T).
    - Learned via Alternating Least Squares (ALS).
    """
<<<<<<< HEAD
    def __init__(self, U_G, U_P_T, K_T1, K_T2, num_classes=None):
=======
    def __init__(self, U_G, U_P_T, K_T1, K_T2):
>>>>>>> 394c7155ace262f89db83c8bdec54dae595b5c33
        self.U_G = U_G
        self.U_P_T = U_P_T # Full temporal GFT basis
        self.N_spatial = U_G.shape[0]
        self.N_temporal = U_P_T.shape[0]
        
        self.K_T1 = K_T1 # Num basis functions for h_hat_1's temporal part
        self.K_T2 = K_T2 # Num basis functions for H_hat_2_T

        if self.K_T1 > self.N_temporal:
            raise ValueError(f"K_T1 ({self.K_T1}) cannot exceed N_temporal ({self.N_temporal})")
        if self.K_T2 > self.N_temporal:
            raise ValueError(f"K_T2 ({self.K_T2}) cannot exceed N_temporal ({self.N_temporal})")

        self.U_P_T_basis1 = self.U_P_T[:, :self.K_T1] # N_T x K_T1
        self.U_P_T_basis2 = self.U_P_T[:, :self.K_T2] # N_T x K_T2

        self.U_ST = np.kron(self.U_G, self.U_P_T) # Full ST GFT basis
        self.N_effective = self.N_spatial * self.N_temporal

        # Model parameters (coefficients)
        self.h0 = np.zeros(self.N_effective,)
        self.h_hat_1_coeffs = np.zeros((self.N_spatial, self.K_T1)) # N_S x K_T1
        self.H_hat_2_S = np.zeros((self.N_spatial, self.N_spatial))
        self.H_hat_2_T_coeffs = np.zeros((self.K_T2, self.K_T2)) # K_T2 x K_T2

<<<<<<< HEAD
        self.num_classes = num_classes
        if num_classes is not None:
            self.classification_weights = np.random.randn(num_classes, self.N_effective) * 0.01
            self.classification_bias = np.random.randn(num_classes) * 0.01
        

=======
>>>>>>> 394c7155ace262f89db83c8bdec54dae595b5c33
    def _reconstruct_full_h_hat_1(self):
        """Reconstructs full h_hat_1 (N_eff,) from h_hat_1_coeffs."""
        # h_hat_1_coeffs is N_S x K_T1
        # For each spatial freq p, temporal filter is sum_k coeffs[p,k] * U_P_T_basis1[:,k]
        # This is h_hat_1_coeffs @ self.U_P_T_basis1.T, giving N_S x N_T matrix
        h_hat_1_matrix_form = self.h_hat_1_coeffs @ self.U_P_T_basis1.T # N_S x N_T
        # Flatten consistent with U_ST = U_G (x) U_P_T (spatial index slower)
        return h_hat_1_matrix_form.flatten(order='C') # if U_ST = U_G (x) U_P_T and s_eff = s_raw.flatten(order='F')
                                                    # then h_hat_1 should be flattened with order 'F'
                                                    # Let's assume s_eff is flattened such that spatial index is outer loop.
                                                    # If s_eff[idx_s*N_T + idx_t], then U_ST = U_G (x) U_P_T
                                                    # and h_hat_1 = h_S (x) h_T.
                                                    # h_hat_1_matrix_form[p,alpha] is filter val at (p,alpha)
                                                    # So flatten with 'C' for row-major (p varies slower)
                                                    # or 'F' for col-major (alpha varies slower)
                                                    # U_ST = U_G kron U_P_T implies (p,alpha) maps to p*N_T + alpha if U_G, U_P_T are indexed 0..N-1
                                                    # So, h_hat_1_matrix_form.flatten(order='C') is correct.
        return h_hat_1_matrix_form.flatten(order='C')


    def _reconstruct_full_H_hat_2_T(self):
        """Reconstructs full H_hat_2_T (N_T x N_T) from H_hat_2_T_coeffs."""
        # H_hat_2_T_coeffs is K_T2 x K_T2
        # H_hat_2_T = U_P_T_basis2 @ H_hat_2_T_coeffs @ U_P_T_basis2.T
        return self.U_P_T_basis2 @ self.H_hat_2_T_coeffs @ self.U_P_T_basis2.T

    def _get_effective_H_hat_2(self):
        """Constructs the full H_hat_2 (N_eff x N_eff) from separable components."""
        full_H_hat_2_T = self._reconstruct_full_H_hat_2_T()
        return np.kron(self.H_hat_2_S, full_H_hat_2_T)

    def set_parameters(self, h0, h_hat_1_coeffs, H_hat_2_S, H_hat_2_T_coeffs):
        self.h0 = h0.reshape(self.N_effective,)
        self.h_hat_1_coeffs = h_hat_1_coeffs.reshape(self.N_spatial, self.K_T1)
        self.H_hat_2_S = H_hat_2_S.reshape(self.N_spatial, self.N_spatial)
        self.H_hat_2_T_coeffs = H_hat_2_T_coeffs.reshape(self.K_T2, self.K_T2)

<<<<<<< HEAD
    def forward(self, s_effective, apply_linear=False):
=======
    def forward(self, s_effective):
>>>>>>> 394c7155ace262f89db83c8bdec54dae595b5c33
        s_col = s_effective.reshape(self.N_effective, 1)
        z0 = self.h0.reshape(self.N_effective, 1)

        s_hat = gft(s_col, self.U_ST)
        full_h_hat_1 = self._reconstruct_full_h_hat_1().reshape(self.N_effective, 1)
        z1_hat = full_h_hat_1 * s_hat 
        z1 = igft(z1_hat, self.U_ST)

        H2_effective = self._get_effective_H_hat_2()
        s_hat_outer_prod = s_hat @ s_hat.T
        Y_hat_prod = H2_effective * s_hat_outer_prod
        Y_prod = self.U_ST @ Y_hat_prod @ self.U_ST.T
        z2 = np.diag(Y_prod).reshape(self.N_effective, 1)

<<<<<<< HEAD
        output_signal = (z0 + z1 + z2).flatten()

        if self.num_classes is not None and apply_linear:
            return self.classification_weights @ output_signal + self.classification_bias

        return output_signal
=======
        return (z0 + z1 + z2).flatten()
>>>>>>> 394c7155ace262f89db83c8bdec54dae595b5c33

    def _build_phi_h0(self):
        return np.eye(self.N_effective)

    def _build_phi_h1_coeffs(self, s_hat_sample_flat_vector): # N_eff vector
        """Builds feature matrix for h_hat_1_coeffs (N_S * K_T1 params)."""
        # h_hat_1_coeffs are N_S x K_T1. We learn vec(h_hat_1_coeffs)
        # Full h_hat_1 = vec(coeffs @ U_P_T_basis1.T) (flattened 'C' for N_S x N_T)
        # z1 = U_ST @ diag(full_h_hat_1) @ s_hat_sample_flat_vector
        #    = U_ST @ diag(s_hat_sample_flat_vector) @ full_h_hat_1
        # full_h_hat_1 is linear in vec(coeffs).
        # full_h_hat_1_alpha_p = sum_k coeffs_pk * U_P_T_basis1_alpha_k
        # Let C = h_hat_1_coeffs. vec(C @ B.T) = (B @ I_KT1) vec(C) if C is N_S x K_T1, B is N_T x K_T1
        # Or, vec(C @ B.T) = (kron(I_NS, B)) vec(C.T)
        # Let's build features for each c_pk coefficient.
        # c_pk contributes to h_hat_1_matrix_form[p, :] via U_P_T_basis1[:, k]
        # So h_hat_1_matrix_form[p, alpha] gets c_pk * U_P_T_basis1[alpha, k]
        
        phi_cols = []
        s_hat_diag_U_ST_T = np.diag(s_hat_sample_flat_vector) @ self.U_ST.T # N_eff x N_eff

        for p_idx in range(self.N_spatial):
            for k_idx in range(self.K_T1):
                # Construct the full_h_hat_1 vector if only c_pk = 1 and others = 0
                h_hat_1_contrib_matrix = np.zeros((self.N_spatial, self.N_temporal))
                h_hat_1_contrib_matrix[p_idx, :] = self.U_P_T_basis1[:, k_idx]
                h_hat_1_contrib_flat = h_hat_1_contrib_matrix.flatten(order='C') # N_eff
                
                # z1 contribution = U_ST @ diag(s_hat) @ h_hat_1_contrib_flat
                # Or = (s_hat_diag_U_ST_T @ h_hat_1_contrib_flat).T if U_ST is real.
                # More directly: U_ST @ np.diag(h_hat_1_contrib_flat) @ s_hat_sample_flat_vector
                # Or: (U_ST * h_hat_1_contrib_flat[:, np.newaxis]) @ s_hat_sample_flat_vector -> incorrect
                # Correct: U_ST @ (h_hat_1_contrib_flat * s_hat_sample_flat_vector).reshape(-1,1)
                # Let's use the form z1 = U_ST diag(s_hat) full_h_hat_1
                # Feature column is U_ST diag(s_hat) (derivative of full_h_hat_1 w.r.t c_pk)
                # Derivative of full_h_hat_1 w.r.t c_pk is the h_hat_1_contrib_flat vector itself.
                
                feature_col = self.U_ST @ np.diag(s_hat_sample_flat_vector) @ h_hat_1_contrib_flat
                phi_cols.append(feature_col)
        return np.array(phi_cols).T # N_eff x (N_S * K_T1)


    def _build_phi_H2_S(self, s_hat_outer_prod_flat_vector, current_full_H_hat_2_T):
        # This is the same as before, just uses the reconstructed current_full_H_hat_2_T
        phi_H2_S_cols = []
        for p_idx in range(self.N_spatial):
            for q_idx in range(self.N_spatial):
                E_pq_S = np.zeros((self.N_spatial, self.N_spatial))
                E_pq_S[p_idx, q_idx] = 1.0
                K_term = np.kron(E_pq_S, current_full_H_hat_2_T)
                Q_mat_elements = K_term.flatten() * s_hat_outer_prod_flat_vector 
                Q_mat = Q_mat_elements.reshape(self.N_effective, self.N_effective)
                Y_prod_contrib = self.U_ST @ Q_mat @ self.U_ST.T
                phi_H2_S_cols.append(np.diag(Y_prod_contrib))
        return np.array(phi_H2_S_cols).T

    def _build_phi_H2_T_coeffs(self, s_hat_outer_prod_flat_vector, current_H_hat_2_S):
        """Builds feature matrix for H_hat_2_T_coeffs (K_T2 * K_T2 params)."""
        phi_H2_T_coeffs_cols = []
        # H_hat_2_T_coeffs is K_T2 x K_T2. We learn vec(H_hat_2_T_coeffs).
        # Full H_hat_2_T = U_basis2 @ Coeffs @ U_basis2.T
        # For each element c_ij in Coeffs (K_T2 x K_T2):
        # Its contribution to Full H_hat_2_T is U_basis2[:,i] @ U_basis2[:,j].T (outer product scaled by c_ij)
        
        for r_idx in range(self.K_T2): # row index in H_hat_2_T_coeffs
            for c_idx in range(self.K_T2): # col index in H_hat_2_T_coeffs
                # Construct Full H_hat_2_T if only H_hat_2_T_coeffs[r,c] = 1, others 0
                H_T_coeff_basis_element = np.zeros((self.K_T2, self.K_T2))
                H_T_coeff_basis_element[r_idx, c_idx] = 1.0
                
                H_hat_2_T_contrib = self.U_P_T_basis2 @ H_T_coeff_basis_element @ self.U_P_T_basis2.T # N_T x N_T
                
                # Now use this H_hat_2_T_contrib in the usual way for z2 feature
                K_term = np.kron(current_H_hat_2_S, H_hat_2_T_contrib) # N_eff x N_eff
                Q_mat_elements = K_term.flatten() * s_hat_outer_prod_flat_vector
                Q_mat = Q_mat_elements.reshape(self.N_effective, self.N_effective)
                Y_prod_contrib = self.U_ST @ Q_mat @ self.U_ST.T
                phi_H2_T_coeffs_cols.append(np.diag(Y_prod_contrib))

        return np.array(phi_H2_T_coeffs_cols).T # N_eff x (K_T2^2)

<<<<<<< HEAD
    def fit(self, S_train_effective, Y_train_classes, num_als_iterations=5, l2_reg=1e-6):
        if self.num_classes is None:
            raise ValueError("num_classes must be specified for classification training.")

        num_samples_fit = S_train_effective.shape[0]
        Y_target_flat = Y_train_classes  # One-hot class labels, shape: (num_samples_fit, num_classes)

        print("Pre-calculating GFTs of training samples...")
        S_hat_train_flat_all = np.array([
            gft(s.reshape(self.N_effective, 1), self.U_ST).flatten() 
            for s in S_train_effective
        ])

        # Initialize Model Parameters
=======

    def fit(self, S_train_effective, Y_train_effective, num_als_iterations=5, l2_reg=1e-6):
        num_samples_fit = S_train_effective.shape[0]
        Y_target_flat = Y_train_effective.reshape(-1, 1)

        print("Pre-calculating GFTs of training samples...")
        S_hat_train_flat_all = np.array([gft(s.reshape(self.N_effective,1), self.U_ST).flatten() for s in S_train_effective])

        # Initialize parameters
>>>>>>> 394c7155ace262f89db83c8bdec54dae595b5c33
        self.h0 = np.random.randn(self.N_effective) * 0.01
        self.h_hat_1_coeffs = np.random.randn(self.N_spatial, self.K_T1) * 0.01
        self.H_hat_2_S = np.random.randn(self.N_spatial, self.N_spatial) * 0.01
        self.H_hat_2_T_coeffs = np.random.randn(self.K_T2, self.K_T2) * 0.01
        self.H_hat_2_S = (self.H_hat_2_S + self.H_hat_2_S.T) / 2
<<<<<<< HEAD
        self.H_hat_2_T_coeffs = (self.H_hat_2_T_coeffs + self.H_hat_2_T_coeffs.T) / 2
        

        self.classification_weights = np.random.randn(self.num_classes, self.N_effective) * 0.01
        self.classification_bias = np.random.randn(self.num_classes) * 0.01
=======
        self.H_hat_2_T_coeffs = (self.H_hat_2_T_coeffs + self.H_hat_2_T_coeffs.T) / 2 # Symmetry for coeffs
>>>>>>> 394c7155ace262f89db83c8bdec54dae595b5c33

        print(f"--- Starting ALS for {num_als_iterations} iterations ---")
        for als_iter in range(num_als_iterations):
            iter_time_start = time.time()
            print(f"\nALS Iteration {als_iter + 1}/{num_als_iterations}")

<<<<<<< HEAD
            # 1. Compute Volterra Feature Representations
            X_features = np.array([
                self.forward(S_train_effective[i]) 
                for i in range(num_samples_fit)
            ])  # Shape: (num_samples_fit, N_effective)

            # 2. Solve for Classification Layer (Linear Least Squares)
            X_aug = np.hstack([X_features, np.ones((X_features.shape[0], 1))])  # Add bias column
            A_cls = X_aug.T @ X_aug + l2_reg * np.eye(X_aug.shape[1])
            B_cls = X_aug.T @ Y_target_flat
            params_cls, _, _, _ = np.linalg.lstsq(A_cls, B_cls, rcond=None)

            self.classification_weights = params_cls[:-1, :].T  # Shape: (num_classes, N_effective)
            self.classification_bias = params_cls[-1, :]

            # 3. Evaluate Classification Performance (Optional)
            logits = (X_features @ self.classification_weights.T) + self.classification_bias
            preds = np.argmax(logits, axis=1)
            true_labels = np.argmax(Y_train_classes, axis=1)
            acc = np.mean(preds == true_labels)
            iter_time_end = time.time()
            print(f"  Iteration {als_iter + 1} Classification Accuracy: {acc:.4f}. Time: {iter_time_end - iter_time_start:.2f}s")

        print("\n--- ALS classification-only fitting complete ---")

=======
            # --- 1. Solve for h0 and h_hat_1_coeffs ---
            # Target: Y - z2(current H2S, H2T_coeffs)
            print(f"  Solving for h0, h_hat_1_coeffs (N_params = {self.N_effective + self.N_spatial * self.K_T1})...")
            current_H2_eff = self._get_effective_H_hat_2()
            Y_target_for_h0h1_list = []
            for i in range(num_samples_fit):
                s_hat_col = S_hat_train_flat_all[i,:].reshape(self.N_effective,1)
                s_hat_outer_prod = s_hat_col @ s_hat_col.T
                Y_hat_prod_i = current_H2_eff * s_hat_outer_prod
                Y_prod_i = self.U_ST @ Y_hat_prod_i @ self.U_ST.T
                z2_i = np.diag(Y_prod_i)
                Y_target_for_h0h1_list.append(Y_train_effective[i,:] - z2_i)
            Y_target_for_h0h1_flat = np.array(Y_target_for_h0h1_list).reshape(-1,1)

            Phi_h0h1_list = []
            for i in range(num_samples_fit):
                s_hat_sample_flat_vec = S_hat_train_flat_all[i,:]
                phi_h0_i = self._build_phi_h0()
                phi_h1_coeffs_i = self._build_phi_h1_coeffs(s_hat_sample_flat_vec)
                Phi_h0h1_list.append(np.hstack([phi_h0_i, phi_h1_coeffs_i]))
            
            Phi_h0h1_step = np.vstack(Phi_h0h1_list)
            num_params_h0h1_step = Phi_h0h1_step.shape[1]
            
            A_h0h1 = Phi_h0h1_step.T @ Phi_h0h1_step + l2_reg * np.eye(num_params_h0h1_step)
            b_h0h1 = Phi_h0h1_step.T @ Y_target_for_h0h1_flat
            params_h0h1_sol, _, _, _ = np.linalg.lstsq(A_h0h1, b_h0h1, rcond=None)
            
            self.h0 = params_h0h1_sol[:self.N_effective].flatten()
            self.h_hat_1_coeffs = params_h0h1_sol[self.N_effective:].reshape(self.N_spatial, self.K_T1)

            # --- 2. Solve for H_hat_2_S ---
            # Target: Y - z0(current h0) - z1(current h1_coeffs)
            print(f"  Solving for H_hat_2_S (N_params = {self.N_spatial**2})...")
            current_full_h_hat_1 = self._reconstruct_full_h_hat_1().reshape(self.N_effective,1)
            Y_target_for_H2S_list = []
            for i in range(num_samples_fit):
                s_hat_col = S_hat_train_flat_all[i,:].reshape(self.N_effective,1)
                z0_i = self.h0
                z1_hat_i = current_full_h_hat_1 * s_hat_col
                z1_i = igft(z1_hat_i, self.U_ST).flatten()
                Y_target_for_H2S_list.append(Y_train_effective[i,:] - z0_i - z1_i)
            Y_target_for_H2S_flat = np.array(Y_target_for_H2S_list).reshape(-1,1)
            
            current_full_H_hat_2_T = self._reconstruct_full_H_hat_2_T()
            Phi_H2S_list = []
            for i in range(num_samples_fit):
                s_h_col_vec = S_hat_train_flat_all[i,:].reshape(self.N_effective, 1)
                s_h_outer_prod_flat_vec = (s_h_col_vec @ s_h_col_vec.T).flatten()
                Phi_H2S_list.append(self._build_phi_H2_S(s_h_outer_prod_flat_vec, current_full_H_hat_2_T))
            
            Phi_H2S_step = np.vstack(Phi_H2S_list)
            num_params_H2S_step = Phi_H2S_step.shape[1]

            A_H2S = Phi_H2S_step.T @ Phi_H2S_step + l2_reg * np.eye(num_params_H2S_step)
            b_H2S = Phi_H2S_step.T @ Y_target_for_H2S_flat
            params_H2S_sol, _, _, _ = np.linalg.lstsq(A_H2S, b_H2S, rcond=None)
            self.H_hat_2_S = params_H2S_sol.reshape(self.N_spatial, self.N_spatial)

            # --- 3. Solve for H_hat_2_T_coeffs ---
            # Target: Y - z0(current h0) - z1(current h1_coeffs) (same target as for H2S)
            print(f"  Solving for H_hat_2_T_coeffs (N_params = {self.K_T2**2})...")
            # Y_target_for_H2T_flat is same as Y_target_for_H2S_flat

            Phi_H2Tcoeffs_list = []
            for i in range(num_samples_fit):
                s_h_col_vec = S_hat_train_flat_all[i,:].reshape(self.N_effective, 1)
                s_h_outer_prod_flat_vec = (s_h_col_vec @ s_h_col_vec.T).flatten()
                Phi_H2Tcoeffs_list.append(self._build_phi_H2_T_coeffs(s_h_outer_prod_flat_vec, self.H_hat_2_S)) # Use updated H2S
            
            Phi_H2Tcoeffs_step = np.vstack(Phi_H2Tcoeffs_list)
            num_params_H2Tcoeffs_step = Phi_H2Tcoeffs_step.shape[1]

            A_H2Tcoeffs = Phi_H2Tcoeffs_step.T @ Phi_H2Tcoeffs_step + l2_reg * np.eye(num_params_H2Tcoeffs_step)
            b_H2Tcoeffs = Phi_H2Tcoeffs_step.T @ Y_target_for_H2S_flat # Using same target
            params_H2Tcoeffs_sol, _, _, _ = np.linalg.lstsq(A_H2Tcoeffs, b_H2Tcoeffs, rcond=None)
            self.H_hat_2_T_coeffs = params_H2Tcoeffs_sol.reshape(self.K_T2, self.K_T2)
            
            # --- Iteration complete, calculate SSR ---
            Y_pred_current_flat = np.array([self.forward(s) for s in S_train_effective]).flatten()
            current_ssr = np.sum((Y_target_flat.flatten() - Y_pred_current_flat)**2)
            iter_time_end = time.time()
            print(f"  ALS Iteration {als_iter + 1} complete. SSR: {current_ssr:.4e}. Time: {iter_time_end - iter_time_start:.2f}s")
        print("\n--- ALS fitting complete ---")
>>>>>>> 394c7155ace262f89db83c8bdec54dae595b5c33

# --- Example Usage ---
if __name__ == '__main__':
    N_spatial_nodes = 10    
    N_temporal_nodes = 20 # Actual N_T
    K_T1_user = 5          # Number of temporal basis functions for h1
    K_T2_user = 5          # Number of temporal basis functions for H2_T (forms K_T2 x K_T2 coeff matrix)
    
    sample_factor = 20 
    num_als_iterations_user = 3
    l2_reg_user = 1e-4

    N_effective_nodes = N_spatial_nodes * N_temporal_nodes
    total_script_start_time = time.time()

    print(f"--- Spatio-Temporal Graph Volterra (Temporal Basis & ALS) ---")
    print(f"Spatial N: {N_spatial_nodes}, Temporal T: {N_temporal_nodes}, N_eff: {N_effective_nodes}")
    print(f"K_T1 (h1 basis): {K_T1_user}, K_T2 (H2T basis): {K_T2_user}")
    print(f"ALS Iterations: {num_als_iterations_user}, L2 Reg: {l2_reg_user}, Sample Factor: {sample_factor}")

    np.random.seed(42)
    adj_G = (np.random.rand(N_spatial_nodes, N_spatial_nodes) < 0.6).astype(float)
    np.fill_diagonal(adj_G, 0); adj_G = (adj_G + adj_G.T)/2; adj_G[adj_G > 0.1] = 1 # ensure binary and symmetric

    L_G = get_combinatorial_laplacian(adj_G)
    eigvals_G, U_G = eigh(L_G); sort_idx_G = np.argsort(eigvals_G); U_G = U_G[:, sort_idx_G]
    
    L_P_T = get_path_graph_laplacian(N_temporal_nodes)
    eigvals_P_T, U_P_T = eigh(L_P_T); sort_idx_P_T = np.argsort(eigvals_P_T); U_P_T = U_P_T[:, sort_idx_P_T]
    
    model = GraphVolterraModelTemporalBasis(U_G, U_P_T, K_T1_user, K_T2_user)

    # Define true parameters (coefficients for the basis)
    true_h0 = np.random.randn(N_effective_nodes,) * 0.05
    true_h_hat_1_coeffs = np.random.randn(N_spatial_nodes, K_T1_user) * 0.1
    true_H_S_temp = np.random.randn(N_spatial_nodes, N_spatial_nodes) * 0.2
    true_H_hat_2_S = (true_H_S_temp + true_H_S_temp.T) / 2 
    true_H_T_coeffs_temp = np.random.randn(K_T2_user, K_T2_user) * 0.2
    true_H_hat_2_T_coeffs = (true_H_T_coeffs_temp + true_H_T_coeffs_temp.T) / 2

    true_model_for_data_gen = GraphVolterraModelTemporalBasis(U_G, U_P_T, K_T1_user, K_T2_user)
    true_model_for_data_gen.set_parameters(true_h0, true_h_hat_1_coeffs, true_H_hat_2_S, true_H_hat_2_T_coeffs)

    num_params_h0 = N_effective_nodes
    num_params_h1coeffs = N_spatial_nodes * K_T1_user
    num_params_H2S = N_spatial_nodes**2
    num_params_H2Tcoeffs = K_T2_user**2
    num_params_total = num_params_h0 + num_params_h1coeffs + num_params_H2S + num_params_H2Tcoeffs
    num_train_samples = int(sample_factor * num_params_total)
    
    min_params_in_als_h0h1 = num_params_h0 + num_params_h1coeffs
    min_params_in_als_H2S = num_params_H2S
    min_params_in_als_H2Tcoeffs = num_params_H2Tcoeffs
    
    required_min_samples_h0h1 = int(np.ceil(min_params_in_als_h0h1 / N_effective_nodes)) + 5
    required_min_samples_H2S = int(np.ceil(min_params_in_als_H2S / N_effective_nodes)) + 5
    required_min_samples_H2Tcoeffs = int(np.ceil(min_params_in_als_H2Tcoeffs / N_effective_nodes)) + 5
    num_train_samples = max(num_train_samples, required_min_samples_h0h1, required_min_samples_H2S, required_min_samples_H2Tcoeffs)
    
    print(f"Total parameters (basis): {num_params_total} (h0:{num_params_h0}, h1c:{num_params_h1coeffs}, HSc:{num_params_H2S}, HTc:{num_params_H2Tcoeffs})")
    print(f"Target training samples: {num_train_samples}")

    S_train_st_raw = np.random.randn(num_train_samples, N_spatial_nodes, N_temporal_nodes)
    S_train_st_effective = np.array([s.flatten(order='C') for s in S_train_st_raw]) # order 'C' to match h_hat_1 reconstruction
    Y_train_st_effective = np.zeros((num_train_samples, N_effective_nodes))

    print("Generating training data...")
    for i in range(num_train_samples):
        Y_train_st_effective[i, :] = true_model_for_data_gen.forward(S_train_st_effective[i, :])
    
    model.fit(S_train_st_effective, Y_train_st_effective, 
              num_als_iterations=num_als_iterations_user, l2_reg=l2_reg_user)

    print("\n--- Parameter Comparison (Relative Norm Difference) ---")
    epsilon = 1e-9
    print(f"  h0 error: {np.linalg.norm(true_h0 - model.h0) / (np.linalg.norm(true_h0) + epsilon):.4e}")
    print(f"  h1_coeffs error: {np.linalg.norm(true_h_hat_1_coeffs - model.h_hat_1_coeffs) / (np.linalg.norm(true_h_hat_1_coeffs) + epsilon):.4e}")
    print(f"  H2_S error: {np.linalg.norm(true_H_hat_2_S - model.H_hat_2_S) / (np.linalg.norm(true_H_hat_2_S) + epsilon):.4e}")
    print(f"  H2_T_coeffs error: {np.linalg.norm(true_H_hat_2_T_coeffs - model.H_hat_2_T_coeffs) / (np.linalg.norm(true_H_hat_2_T_coeffs) + epsilon):.4e}")

    s_test_raw = np.random.randn(N_spatial_nodes, N_temporal_nodes)
    s_test_eff = s_test_raw.flatten(order='C')
    y_true_test = true_model_for_data_gen.forward(s_test_eff)
    y_pred_test = model.forward(s_test_eff)
    pred_err = np.linalg.norm(y_true_test - y_pred_test) / (np.linalg.norm(y_true_test) + epsilon)
    print(f"\nTest Prediction Error (Relative Norm Diff): {pred_err:.4e}")
    
    total_script_end_time = time.time()
    print(f"\n--- Total script execution time: {total_script_end_time - total_script_start_time:.2f} seconds ---")
<<<<<<< HEAD
=======

>>>>>>> 394c7155ace262f89db83c8bdec54dae595b5c33
