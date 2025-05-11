import torch 
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

def train_epoch(model, loader, opt, device):
    model.train()
    total = 0
    for data in loader:
        data = data.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(data), data.y.view(-1))
        loss.backward()
        opt.step()
        total += loss.item() * data.num_graphs
    return total / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            preds = model(data).argmax(dim=1)
            correct += (preds == data.y.view(-1)).sum().item()
            total   += data.num_graphs
    return correct / total
