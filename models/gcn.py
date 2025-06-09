import torch
import torch.nn as nn
import torch.nn.functional as f
import torch_geometric.nn as pyg_nn
from typing import Literal

class GCN(nn.Module):
    """
    Graph convolutional network.
    
    This model applies a stack of GCN layers followed by global pooling and
    a final linear classifier. Optional components include edge weights, 
    batch normalization, and flexible pooling strategies.
    """

    def __init__(
        self,
        num_layers: int,
        nfeat: int,
        nhid: int,
        nclass: int,
        dropout: float,
        adj_weight: bool = False,
        use_bn: bool = False,
        pool: Literal["mean", "max"] = "mean"
    ):
        """Initialize the GCN model.

        Args:
            num_layers (int): Number of layers (excluding the final projection layer).
            nfeat (int): Number of input features.
            nhid (int): Number of hidden units per layer.
            nclass (int): Number of output classes.
            dropout (float): Dropout rate used after each layer (0-1).
            adj_weight (bool, optional): Whether to use edge weights during GCN propagation. False by defalut.
            use_bn (bool, optional): Whether to apply batch normalization after each layer. False by defalut.
            pool (Literal["mean", "max"], optional): Pooling strategy for graph-level readout. Defaults to "mean".
        """
        super(GCN, self).__init__()

        self.input_channels = nfeat
        self.output_channels = nclass
        self.hidden_channels = nhid
        
        self.dropout = dropout
        self.activation = f.relu
        self.use_bn = use_bn
        self.weighted = adj_weight

        #Creating stack of layers
        self.convs = nn.ModuleList()
        self.convs.append(
            pyg_nn.GCNConv(
                self.input_channels,
                self.hidden_channels,
            )
        )
        self.bns = nn.ModuleList()
        self.bns.append(pyg_nn.LayerNorm(self.hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(
                pyg_nn.GCNConv(
                    self.hidden_channels,
                    self.hidden_channels,
                )
            )
            self.bns.append(pyg_nn.LayerNorm(self.hidden_channels))
        
        self.conv_last = pyg_nn.GCNConv(
            self.hidden_channels,
            self.hidden_channels,
        )

        self.linear = nn.Linear(self.hidden_channels, self.output_channels)
        
        #Pooling strategy
        if pool == "mean":
          self.pool = pyg_nn.global_mean_pool
        elif pool == "max":
          self.pool = pyg_nn.global_max_pool

    
    def forward(self, data) -> torch.Tensor:
        """Forward pass of GCN model.

        Args:
            data: Graph data.

        Returns:
            The output of the model.
        """
        x, ei, ew, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        for i, conv in enumerate(self.convs):
            if self.weighted:
                x = conv(x, ei, edge_weight=ew)
            else: 
                x = conv(x, ei)
                
            if self.use_bn:
                x = self.bns[i](x, batch = batch)
                
            x = self.activation(x)
            x = f.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv_last(x, ei)
        x = self.pool(x, batch)
        x = self.linear(x)
        
        return f.softmax(x, dim=1)
