import torch
import torch.nn as nn
import torch.nn.functional as f
import torch_geometric.nn as pyg_nn


class GCN(nn.Module):
    """Graph convolutional network."""

    def __init__(
        self,
        num_layers: int,
        nfeat: int,
        nhid: int,
        nclass: int,
        dropout: float,
        use_bn: bool = False,
    ):
        """Initialize the model.

        Args:
            num_layers: Number of layers.
            nfeat: Number of input features.
            nhid: Number of hidden features.
            nclass: Number of output features.
            dropout: Dropout rate.
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
        self.bns.append(nn.BatchNorm1d(self.hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(
                pyg_nn.GCNConv(
                    self.hidden_channels,
                    self.hidden_channels,
                )
            )
            self.bns.append(nn.BatchNorm1d(self.hidden_channels))

        self.dropout = dropout
        self.activation = f.relu
        self.use_bn = use_bn

        self.linear = nn.Linear(self.hidden_channels, self.output_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: The input features.
            edge_index: The edge index.

        Returns:
            The output of the model.
        """
        x = f.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = f.dropout(x, p=self.dropout, training=self.training)
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
        use_bn: bool = False,
    ):
        """Initialize the model.

        Args:
            num_layers: Number of layers.
            nfeat: Number of input features.
            nhid: Number of hidden features.
            nclass: Number of output features.
            dropout: Dropout rate.
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
        self.bns.append(nn.BatchNorm1d(self.hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(
                pyg_nn.SAGEConv(
                    self.hidden_channels,
                    self.hidden_channels,
                )
            )
            self.bns.append(nn.BatchNorm1d(self.hidden_channels))

        self.dropout = dropout
        self.activation = f.relu
        self.use_bn = use_bn

        self.linear = nn.Linear(self.hidden_channels, self.output_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: The input features.
            edge_index: The edge index.

        Returns:
            The output of the model.
        """
        x = f.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = f.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)
        return f.softmax(x, dim=1)


class GAT(nn.Module):
    """Graph attention network."""

    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nclass: int,
        dropout: float,
        use_bn: bool = False,
        num_layers: int = 2,
        heads: int = 2,
        out_heads: int = 1,
    ):
        """Initialize the model.

        Args:
            nfeat: Number of input features.
            nhid: Number of hidden features.
            nclass: Number of output features.
            dropout: Dropout rate.
            use_bn: Whether to use batch normalization.
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
        self.bns.append(nn.BatchNorm1d(self.hidden_channels * heads))
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
            self.bns.append(nn.BatchNorm1d(self.hidden_channels * heads))

        self.convs.append(
            pyg_nn.GATConv(
                self.hidden_channels * heads,
                self.output_channels,
                dropout=dropout,
                heads=out_heads,
                concat=False,
            )
        )

        self.dropout = dropout
        self.activation = f.elu
        self.use_bn = use_bn

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: The input features.
            edge_index: The edge index.

        Returns:
            The output of the model.
        """
        x = f.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = f.dropout(x, p=self.dropout, training=self.training)
        x = self.linear(x)
        return f.softmax(x, dim=1)
