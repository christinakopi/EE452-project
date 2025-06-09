import torch
import torch.nn as nn
import torch.nn.functional as f
import torch_geometric.nn as pyg_nn
from typing import Literal

class GIN(torch.nn.Module):
    """
    Graph Isomorphism Network (GIN) for graph-level classification.

    This implementation uses MLP-based GINConv layers for message passing, with optional batch normalization,
    ReLU activations, and dropout. A global pooling layer aggregates node features to form a graph-level representation
    before classification.
    """
    def __init__(
        self,
        num_layers: int,
        nfeat: int,
        nhid: int,
        nclass: int,
        dropout: float,
        use_bn: bool = False,
        pool: Literal["mean", "max"] = "mean"
        ):
        """Initialize the model.

        Args:
            num_layers (int): Number of layers (excluding the final projection layer).
            nfeat (int): Number of input features.
            nhid (int): Number of hidden units per layer.
            nclass (int): Number of output classes.
            dropout (float): Dropout rate used after each layer (0-1).
            use_bn (bool, optional): Whether to apply batch normalization after each layer. False by defalut.
            pool (Literal["mean", "max"], optional): Pooling strategy for graph-level readout. Defaults to "mean".
        """
        super(GIN, self).__init__()
        self.input_channels = nfeat
        self.output_channels = nclass
        self.hidden_channels = nhid
        self.num_layers = num_layers
        
        self.dropout = dropout
        self.activation = f.relu
        self.use_bn = use_bn
        
        self.linear0 = nn.Linear(self.input_channels, self.hidden_channels)
        self.convs = torch.nn.ModuleList()
        
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels), 
                nn.BatchNorm1d(self.hidden_channels),
                nn.ReLU(),                       
                nn.Linear(self.hidden_channels, self.hidden_channels), 
                nn.BatchNorm1d(self.hidden_channels),
                nn.ReLU())
            self.convs.append(pyg_nn.GINConv(mlp, eps=0, train_eps=False))
        
        self.linear1 = nn.Linear(self.hidden_channels, self.hidden_channels)
        self.linear2 = nn.Linear(self.hidden_channels, self.output_channels)

        if pool == "mean":
          self.pool = pyg_nn.global_mean_pool
        elif pool == "max":
          self.pool = pyg_nn.global_max_pool

    def forward(self, data):
        """Forward pass.

        Args:
            data: Graph data.

        Returns:
            The output of the model.
        """
        x, ei, ew, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = self.linear0(x)
        x = self.activation(x)
        
        x = f.dropout(x, p=self.dropout, training=self.training)
        
        for i in range(self.num_layers):
            x = self.convs[i](x, ei)
        
        x = self.linear1(x)
        x = self.activation(x)
        x = self.pool(x, batch)
        x = self.linear2(x)
        
        return f.softmax(x, dim=1)
