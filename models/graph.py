# graph_networks.py

import torch.nn as nn
import torch_geometric
from .utils import pairwise

class GraphNeuralNetwork(nn.Module):
    """
    A general graph neural network (GNN) class. Implements multiple varieties of
    GNNs from the torch_geometric library.

    Parameters
    ----------
    in_size (int):
        The size of the input features.
    out_size (int):
        The size of the output features.
    variety (str):
        The type of GNN to use. The implemented types are:
            - 'GAT' : Graph Attention Network
            - 'GATv2' : Graph Attention Network v2
            - 'GCN' : Graph Convolutional Network
            - 'TF' : Transformer Network
            - 'RGGC' : Residual Gated Graph Convolution
    hidden_sizes (list):
        The sizes of the hidden layers of the GNN. Defaults to no hidden layers.
    aggr (str):
        The type of aggregation to use. Defaults to 'mean'.
    """
    def __init__(
            self,
            in_size : int,
            out_size : int,
            variety : str,
            hidden_sizes : list = [],
            aggr : str = 'mean',
            ):
        super().__init__()

        # the implemented GNN types
        networks = {
            'gat' : torch_geometric.nn.GATConv,
            'gatv2' : torch_geometric.nn.GATv2Conv,
            'gcn' : torch_geometric.nn.GCNConv,
            'tf' : torch_geometric.nn.TransformerConv,
            'rggc' : torch_geometric.nn.ResGatedGraphConv
            }
        variety = variety.lower()
        if variety not in networks.keys():
            raise ValueError(
                f'{variety} is not a valid GNN type. The valid types are {networks.keys()}'
                )
        network = networks[variety]

        # construct the sizes of the GNN and store the activation function
        features = [in_size] + hidden_sizes + [out_size]
        self.activation = nn.ReLU()
        self.out_size = out_size

        # construct the layers of the GNN
        layers = []
        for l1, l2 in pairwise(features):
            layers.append(
                network(
                    in_channels = l1,
                    out_channels = l2,
                    aggr = aggr,
                    )
                )
        self.layers = nn.ModuleList(layers)


    def forward(self, graphs : torch_geometric.data.Batch):
        """
        The forward pass of the GNN.

        Parameters
        ----------
        graphs (torch_geometric.data.Batch):
            The batch of graphs to be passed through the GNN.

        Returns
        -------
        torch.Tensor
            The output of the GNN.
        """
        num_graphs = graphs.num_graphs
        features = graphs.x
        edges = graphs.edge_index

        for layer in self.layers[:-1]:
            features = self.activation(layer(features, edges))
        features = self.layers[-1](features, edges)
        return features.view(num_graphs, -1, self.out_size)