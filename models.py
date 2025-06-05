import torch
import torch.nn as nn
import torch.nn.functional as f
import torch_geometric.nn as pyg_nn
from typing import Literal

class GCN(nn.Module):
    """Graph convolutional network."""

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
        """Initialize the model.

        Args:
            num_layers: Number of layers.
            nfeat: Number of input features.
            nhid: Number of hidden features.
            nclass: Number of output features.
            dropout: Dropout rate.
            adj_weight: Whether include the edge_weight.
            use_bn: Whether to use batch normalization.
        """
        super(GCN, self).__init__()

        self.input_channels = nfeat
        self.output_channels = nclass
        self.hidden_channels = nhid

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

        self.dropout = dropout
        self.activation = f.relu
        self.use_bn = use_bn
        self.weighted = adj_weight

        self.linear = nn.Linear(self.hidden_channels, self.output_channels)
        if pool == "mean":
          self.pool = pyg_nn.global_mean_pool
        elif pool == "max":
          self.pool = pyg_nn.global_max_pool

    
    def forward(self, data) -> torch.Tensor:
        """Forward pass.

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
            x = f.dropout(x, p=self.dropout, training=self.training)
            x = self.activation(x)
        
        x = self.conv_last(x, ei)
        x = self.pool(x, batch)
        x = self.linear(x)
        return f.softmax(x, dim=1)


class GNNSage(nn.Module):
    """Graph message-passing network."""

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
        """Initialize the model.

        Args:
            num_layers: Number of layers.
            nfeat: Number of input features.
            nhid: Number of hidden features.
            nclass: Number of output features.
            dropout: Dropout rate.
            adj_weight: Whether include the edge_weight.
            use_bn: Whether to use batch normalization.
        """
        super(GNNSage, self).__init__()

        self.input_channels = nfeat
        self.output_channels = nclass
        self.hidden_channels = nhid

        self.convs = nn.ModuleList()
        self.convs.append(
            pyg_nn.SAGEConv(
                self.input_channels,
                self.hidden_channels,
            )
        )
        self.bns = nn.ModuleList()
        self.bns.append(pyg_nn.LayerNorm(self.hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(
                pyg_nn.SAGEConv(
                    self.hidden_channels,
                    self.hidden_channels,
                )
            )
            self.bns.append(pyg_nn.LayerNorm(self.hidden_channels))

        self.conv_last = pyg_nn.SAGEConv(
            self.hidden_channels,
            self.hidden_channels,
        )

        self.dropout = dropout
        self.activation = f.relu
        self.use_bn = use_bn

        self.weighted = adj_weight

        self.linear = nn.Linear(self.hidden_channels, self.output_channels)
        if pool == "mean":
          self.pool = pyg_nn.global_mean_pool
        elif pool == "max":
          self.pool = pyg_nn.global_max_pool

    
    def forward(self, data) -> torch.Tensor:
        """Forward pass.

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
            x = f.dropout(x, p=self.dropout, training=self.training)
            x = self.activation(x)

        x = self.conv_last(x, ei)
        x = self.pool(x, batch)
        x = self.linear(x)
        return f.softmax(x, dim=1)


class GAT(nn.Module):
    """Graph attention network."""

    def __init__(
        self,
        num_layers: int,
        nfeat: int,
        nhid: int,
        nclass: int,
        dropout: float,
        adj_weight: bool = False,
        use_bn: bool = False,
        heads: int = 2,
        out_heads: int = 1,
        pool: Literal["mean", "max"] = "mean"
    ):
        """Initialize the model.

        Args:
            nfeat: Number of input features.
            nhid: Number of hidden features.
            nclass: Number of output features.
            dropout: Dropout rate.
            use_bn: Whether to use batch normalization.
            adj_weight: Whether include the edge_weight.
            num_layers: Number of layers.
            heads: Number of attention heads.
            out_heads: Number of output heads.
        """
        super(GAT, self).__init__()

        self.input_channels = nfeat
        self.output_channels = nclass
        self.hidden_channels = nhid
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(
            pyg_nn.GATConv(
                self.input_channels,
                self.hidden_channels,
                dropout=dropout,
                heads=heads,
                concat=True,
            )
        )

        self.bns = nn.ModuleList()
        self.bns.append(pyg_nn.LayerNorm(self.hidden_channels * heads))
        for _ in range(num_layers - 1):
            self.convs.append(
                pyg_nn.GATConv(
                    self.hidden_channels * heads,
                    self.hidden_channels,
                    dropout=dropout,
                    heads=heads,
                    concat=True,
                )
            )
            self.bns.append(pyg_nn.LayerNorm(self.hidden_channels * heads))

        self.conv_last = pyg_nn.GATConv(
            self.hidden_channels * heads,
            self.hidden_channels,
            dropout=dropout,
            heads=heads,
            concat=True,
        )
        
        self.dropout = dropout
        self.activation = f.elu
        self.use_bn = use_bn
        self.weighted = adj_weight

        self.linear = nn.Linear(self.hidden_channels * heads, self.output_channels)
        if pool == "mean":
          self.pool = pyg_nn.global_mean_pool
        elif pool == "max":
          self.pool = pyg_nn.global_max_pool

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: Graph data.

        Returns:
            The output of the model.
        """
        x, ei, ew, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = f.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs):
            if self.weighted:
                x = conv(x, ei, edge_weight=ew)
            else: 
                x = conv(x, ei)
            if self.use_bn:
                x = self.bns[i](x, batch = batch)
            x = f.dropout(x, p=self.dropout, training=self.training)
            x = self.activation(x)

        x = self.conv_last(x, ei)
        x = self.pool(x, batch)
        x = self.linear(x)
        return f.softmax(x, dim=1)

class GIN(torch.nn.Module):
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
        x, ei, ew, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.linear0(x)
        x = f.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(x)
        for i in range(self.num_layers):
            x = self.convs[i](x, ei)
        
        x = self.linear1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.linear2(x)
        return f.softmax(x, dim=1)
