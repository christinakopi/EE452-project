import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader

class CrossValidator:
    def __init__(self,
                 dataset,
                 model_class,
                 model_kwargs: dict,
                 train_fn,
                 eval_fn,
                 device: torch.device,
                 batch_size: int = 32,
                 epochs: int = 20,
                 learning_rate: float = 1e-3,
                 n_splits: int = 5,
                 shuffle: bool = True,
                 random_state: int = 42):
        """
        dataset: PyG Dataset of graphs with .y labels
        model_class: callable returning a torch.nn.Module
        model_kwargs: dict of kwargs to pass to model_class
        train_fn: function(model, loader, optimizer, device) -> loss (check train.py)
        eval_fn: function(model, loader, device) -> metric (check train.py)
        device: torch.device to use for training
        """
        self.dataset       = dataset
        self.model_class   = model_class
        self.model_kwargs  = model_kwargs
        self.train_fn      = train_fn
        self.eval_fn       = eval_fn
        self.device        = device
        self.batch_size    = batch_size
        self.epochs        = epochs
        self.lr            = learning_rate
        self.n_splits      = n_splits
        self.shuffle       = shuffle
        self.random_state  = random_state

    def run(self):
        # extract labels for stratification
        labels = np.array([data.y.item() for data in self.dataset])
        kf = StratifiedKFold(n_splits=self.n_splits,
                             shuffle=self.shuffle,
                             random_state=self.random_state)
        results = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(np.zeros(len(labels)), labels), 1):
            print(f"\n--- Fold {fold}/{self.n_splits} ---")
            train_ds = torch.utils.data.Subset(self.dataset, train_idx)
            val_ds   = torch.utils.data.Subset(self.dataset, val_idx)
            train_loader = DataLoader(train_ds,
                                      batch_size=self.batch_size,
                                      shuffle=True)
            val_loader   = DataLoader(val_ds,
                                      batch_size=self.batch_size,
                                      shuffle=False)

            model     = self.model_class(**self.model_kwargs).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

            for epoch in range(1, self.epochs+1):
                loss = self.train_fn(model, train_loader, optimizer, self.device)
                if epoch % 5 == 0 or epoch == self.epochs:
                    acc = self.eval_fn(model, val_loader, self.device)
                    print(f"Epoch {epoch}: Val = {acc}")
            final = self.eval_fn(model, val_loader, self.device)
            print(f"Fold {fold} final Val = {final}")
            results.append(final)

        mean = np.mean(results)
        std  = np.std(results)
        print(f"\nCross-validation result: {mean} ± {std}")
        return results