import torch
from torch import nn, Tensor
from typing import List
from functools import reduce

class VolterraGraphConvLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, order: int, adjacency_matrix_powers: List[Tensor]):
        super(VolterraGraphConvLayer, self).__init__()
        
        self.projective_layers = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(order+1)])
        self.activation = nn.ReLU()
        self.adjacency_matrix_powers = adjacency_matrix_powers
        self.order = order
        self.out_dim = out_dim

    def forward(self, X: Tensor) -> Tensor:
        powers = torch.arange(0, len(self.adjacency_matrix_powers), dtype=torch.int64)
        out = self.projective_layers[0](X)

        mat_muls = [torch.matmul(A, X) for A in self.adjacency_matrix_powers]
        
        for r in range(1, self.order+1):
            indices = torch.combinations(powers, r=r, with_replacement=True)
            
            for index_set in indices:
                product = reduce(lambda x, y: x * y, [mat_muls[idx] for idx in index_set])
                out += self.projective_layers[r](product)
        
        return self.activation(out)

class VolterraGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, order: int, adjacency_matrix: Tensor, num_layers: int, num_outputs: int):
        super(VolterraGCN, self).__init__()
        
        self.order = order
        self.num_layers = num_layers
        
        powers = [torch.eye(adjacency_matrix.shape[0])]
        for _ in range(1, order + 1):
            powers.append(torch.matmul(powers[-1], adjacency_matrix))
        
        self.input_layer = VolterraGraphConvLayer(in_dim, hidden_dim, order, powers)
        self.hidden_layers = nn.ModuleList([VolterraGraphConvLayer(hidden_dim, hidden_dim, order, powers) for _ in range(num_layers)])
        self.out_layer = VolterraGraphConvLayer(hidden_dim, num_outputs, order, powers)
    
    def forward(self, X: Tensor) -> Tensor:
        X = self.input_layer(X)
        for layer in self.hidden_layers:
            X = layer(X)
        X = self.out_layer(X)
        return X