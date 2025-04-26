import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
from voltera_gcn import VolterraGCN

dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]

A = to_dense_adj(data.edge_index)[0]
A = torch.eye(len(A))*torch.diag(A) - A
X = data.x
y = data.y

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VolterraGCN(in_dim=X.shape[1], hidden_dim=512, order=1, adjacency_matrix=A, num_layers=4, num_outputs=dataset.num_classes).to(device)
X, A, y = X.to(device), A.to(device), y.to(device)

X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-8)
def normalize_adjacency_matrix(A: torch.Tensor) -> torch.Tensor:
    D = torch.diag(torch.pow(A.sum(1), -0.5))
    D[D == float('inf')] = 0
    A_hat = torch.matmul(torch.matmul(D, A), D)
    return A_hat

A = to_dense_adj(data.edge_index)[0]
A = normalize_adjacency_matrix(A)
A = A / torch.linalg.norm(A, ord=2)


optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

num_epochs = 200

model.train()
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(X)
    
    loss = criterion(output[data.train_mask], y[data.train_mask])
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            model.eval()
            pred = output.argmax(dim=1)
            train_acc = (pred[data.train_mask] == y[data.train_mask]).sum().item() / data.train_mask.sum().item()
            val_acc = (pred[data.val_mask] == y[data.val_mask]).sum().item() / data.val_mask.sum().item()
            print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')


model.eval()
with torch.no_grad():
    output = model(X)
    pred = output.argmax(dim=1)
    test_acc = (pred[data.test_mask] == y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    print(f'Test Accuracy: {test_acc:.4f}')
