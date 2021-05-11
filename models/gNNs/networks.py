import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.spatial
from torch_geometric.nn import knn_graph, SAGPooling, graclus
from dgl.nn.pytorch import GraphConv
from dgl.subgraph import edge_subgraph

from models.gNNs.layers import GNNLayer

class BasicGCNRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(BasicGCNRegressor, self).__init__()
        self.dropout = 0.5
        self.conv1 = GraphConv(in_dim, hidden_dim, activation=nn.ReLU())
        self.conv2 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.conv3 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.conv4 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.conv5 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.conv6 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.predict_layer = nn.Linear(hidden_dim, n_classes)

    def forward(self, graph, features):
        # print("Inside BasicGCNRegressor forward()")
        # Perform graph convolution and activation function.
        hidden = self.conv1(graph, features)
        # print("After self.conv1")
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        hidden = self.conv2(graph, hidden)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        hidden = self.conv3(graph, hidden)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        hidden = self.conv4(graph, hidden)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        hidden = self.conv5(graph, hidden)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        hidden = self.conv6(graph, hidden)
        with graph.local_scope():
            graph.ndata['tmp'] = hidden
            # Calculate graph representation by averaging all the node representations.
            hg = dgl.mean_nodes(graph, 'tmp')

        return self.predict_layer(hg)

class BasicGCNSegmentation(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, device):
        super(BasicGCNSegmentation, self).__init__()
        self.dropout = 0.5
        self.conv1 = GraphConv(in_dim, hidden_dim, activation=nn.ReLU())
        self.conv2 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.conv3 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.conv4 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.conv5 = GraphConv(hidden_dim, n_classes, activation=None)
        # self.sagPooling1 = SAGPooling(hidden_dim)
        # self.sagPooling2 = SAGPooling(hidden_dim)
        self.device = device

    def forward(self, graph, features):
        # Perform graph convolution and activation function.
        hidden = self.conv1(graph, features)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        edge_index = self.knn(hidden, hidden, 20)
        # print("hidden shape")
        # print(hidden.shape)
        # print("edge_index shape")
        # print(edge_index.shape)
        hidden = graclus(edge_index)
        # sag_pool_out1 = self.sagPooling1(hidden.to(self.device), edge_index.to(self.device))
        # # print("sag_pool_out1[0] shape")
        # # print(sag_pool_out1[0].shape)
        # # print("graph")
        # # print(graph)
        # sag_pooled_features = sag_pool_out1[0]
        # sag_edge_index = sag_pool_out1[1]
        # print("graph")
        # print(graph)
        # print("sag_pooled_features shape")
        # print(sag_pooled_features.shape)
        # print("sag_edge_index shape before flattening")
        # print(sag_edge_index.shape)
        # sag_edge_index = torch.flatten(sag_edge_index)
        # print("flattened sag_edge_index shape")
        # print(sag_edge_index.shape)
        # graph = edge_subgraph(graph, sag_edge_index, True)
        # print("subgraph1")
        # print(graph)
        # graph = dgl.add_self_loop(graph)
        # hidden = self.conv2(graph, sag_pooled_features)
        print("graclus hidden shape")
        print(hidden.shape)
        print("graph")
        print(graph)
        hidden = self.conv2(graph.to(self.device), hidden.to(self.device))
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        hidden = self.conv3(graph, hidden)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        hidden = self.conv4(graph, hidden)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        return self.conv5(graph, hidden)

    def knn(self, x, y, k, batch_x=None, batch_y=None):
        if batch_x is None:
            batch_x = x.new_zeros(x.size(0), dtype=torch.long)

        if batch_y is None:
            batch_y = y.new_zeros(y.size(0), dtype=torch.long)

        x = x.view(-1, 1) if x.dim() == 1 else x
        y = y.view(-1, 1) if y.dim() == 1 else y

        assert x.dim() == 2 and batch_x.dim() == 1
        assert y.dim() == 2 and batch_y.dim() == 1
        assert x.size(1) == y.size(1)
        assert x.size(0) == batch_x.size(0)
        assert y.size(0) == batch_y.size(0)

        # Rescale x and y.
        min_xy = min(x.min().item(), y.min().item())
        x, y = x - min_xy, y - min_xy

        max_xy = max(x.max().item(), y.max().item())
        x, y, = x / max_xy, y / max_xy

        # Concat batch/features to ensure no cross-links between examples exist.
        x = torch.cat([x, 2 * x.size(1) * batch_x.view(-1, 1).to(x.dtype)], dim=-1)
        y = torch.cat([y, 2 * y.size(1) * batch_y.view(-1, 1).to(y.dtype)], dim=-1)

        # print("Before calling scipy.spatial.cKDTree()")
        tree = scipy.spatial.cKDTree(x.detach().cpu().numpy(), balanced_tree=False)
        # print("After calling scipy.spatial.cKDTree()")
        # print("Before calling tree.query")
        dist, col = tree.query(
            y.detach().cpu(), k=k, distance_upper_bound=x.size(1))
        # print("After calling tree.query")

        dist = torch.from_numpy(dist).to(x.dtype)
        col = torch.from_numpy(col).to(torch.long)
        row = torch.arange(col.size(0), dtype=torch.long).view(-1, 1).repeat(1, k)
        mask = ~torch.isinf(dist).view(-1)
        row, col = row.view(-1)[mask], col.view(-1)[mask]

        return torch.stack([row, col], dim=0)

class GNNModel(nn.Module):
    def __init__(self, input_node_dim, input_edge_dim, hidden_dim1, hidden_dim2, out_dim):
        super(GNNModel, self).__init__()
        self.gn1 = GNNLayer(input_node_dim, input_edge_dim, hidden_dim1, activation=nn.ReLU())
        self.gc1 = GraphConv(hidden_dim1, hidden_dim2, activation=nn.ReLU())
        self.predict_layer = nn.Linear(hidden_dim2, out_dim)

    def forward(self, g, node_features, edge_features):
        hidden = self.gn1(g, node_features, edge_features)
        hidden = self.gc1(g, hidden)
        with g.local_scope():
            g.ndata['tmp'] = hidden
            # Calculate graph representation by averaging all the node representations.
            hg = dgl.mean_nodes(g, 'tmp')
        return self.predict_layer(hg)
