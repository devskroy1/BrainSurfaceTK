import dgl
import torch
import torch.nn as nn
import scipy.spatial
from torch_geometric.nn import EdgeConv, knn, knn_graph, DynamicEdgeConv
from dgl.nn.pytorch import GraphConv

from models.gNNs.layers import GNNLayer

# if torch.cuda.is_available():
#     import torch_cluster.knn_cuda

# from torch_cluster import knn

class MLP(nn.Module):
    def __init__(self, layer_dims, batch_norm=True, relu=True):
        super().__init__()
        self.layer_dims = layer_dims
        layers = []
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            if relu:
                layers.append(nn.ReLU())
        self.layer = nn.Sequential(*layers)

    def reset_parameters(self):
        reset(self.layer)

    def forward(self, x):
        return self.layer(x)

class EdgeConvGCNSegmentation(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, device):
        super(EdgeConvGCNSegmentation, self).__init__()
        self.edge_conv_dims = [[16, 16, 16], [32, 32, 32], [64, 64]]
        self.edge_convs = self.make_edge_conv_layers_()
        self.conv1 = GraphConv(in_dim, hidden_dim, activation=nn.ReLU())
        self.conv2 = GraphConv(hidden_dim*2, hidden_dim*2, activation=nn.ReLU())
        self.conv3 = GraphConv(hidden_dim*4, hidden_dim*2, activation=nn.ReLU())
        # For 2 EdgeConv layers
        self.conv4 = GraphConv(hidden_dim * 2, n_classes, activation=None)
        # For 3 EdgeConv layers
        # self.conv4 = GraphConv(hidden_dim * 4, n_classes, activation=None)
        self.device = device

    def forward(self, graph, features):
        # Perform graph convolution and activation function.
        #EdgeConv
        edgeConv1 = list(self.edge_convs.children())[0]
        edgeConv2 = list(self.edge_convs.children())[1]
        # edgeConv3 = list(self.edge_convs.children())[2]

        # DynamicEdgeConv
        dynEdgeConv1 = list(self.edge_convs.children())[0]
        dynEdgeConv2 = list(self.edge_convs.children())[1]
        # dynEdgeConv3 = list(self.edge_convs.children())[2]
        hidden = self.conv1(graph, features)
        # print("in_dim")
        # print(3)
        # print("hidden_dim")
        # print(256)
        # print("hidden shape after conv1")
        # print(hidden.shape)
        # print("edgeConv1")
        # print(edgeConv1)
        # print("edgeConv2")
        # print(edgeConv2)

        #EdgeConv
        # print("Before calling knn() 1")
        edge_index = self.knn(hidden, hidden, 20)
        # print("After calling knn1")
        hidden = edgeConv1(hidden.to(self.device), edge_index.to(self.device))

        #DynEdgeConv
        #hidden = dynEdgeConv1(hidden.to(self.device))

        # print("hidden shape after edgeConv1")
        # print(hidden.shape)
        hidden = self.conv2(graph, hidden)
        # print("hidden shape after conv2")
        # print(hidden.shape)

        # # EdgeConv
        # print("Before calling knn() 2")
        edge_index = self.knn(hidden, hidden, 20)
        # print("After calling knn() 2")
        hidden = edgeConv2(hidden.to(self.device), edge_index.to(self.device))

        # DynEdgeConv
        #hidden = dynEdgeConv2(hidden.to(self.device))

        # print("hidden shape after edgeConv2")
        # print(hidden.shape)
        hidden = self.conv3(graph, hidden)
        # edge_index = self.knn(hidden, hidden, 20)
        # hidden = edgeConv3(hidden.to(self.device), edge_index.to(self.device))
        hidden = self.conv4(graph, hidden)

        return hidden

    def make_edge_conv_layers_(self):
        """Define structure of the EdgeConv Blocks
        edge_conv_dims: [[convi_mlp_dims]], e.g., [[3, 64], [64, 128]]
        """
        print("Inside make_edge_conv_layers_()")

        layers = []
        for dims in self.edge_conv_dims:
            # mlp_dims = [dims[0] * 2] + dims[1::]
            # mlp_dims = dims
            #mlp_dims = [256, 256, 256]
            layers.append(EdgeConv(nn=MLP(dims), aggr='max'))
            #layers.append(DynamicEdgeConv(nn=MLP(dims), k=50, aggr='max'))
        return nn.Sequential(*layers)

    # def knn(self, x, k):
    #     print("x shape")
    #     print(x.shape)
    #     inner = -2 * torch.matmul(x.transpose(0, 1), x)
    #     #inner shape: 64 x 64
    #     xx = torch.sum(x ** 2, dim=0, keepdim=True)
    #     #xx shape: 64 x 1
    #     print("inner shape")
    #     print(inner.shape)
    #     print("xx.shape")
    #     print(xx.shape)
    #
    #     # pairwise_distance = -xx - inner - xx.transpose(0, 1)
    #     pairwise_distance = -xx - inner
    #
    #     idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    #     return idx

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
            y.detach().cpu(), k=k, distance_upper_bound=x.size(1), n_jobs=-1)
        # print("After calling tree.query")

        dist = torch.from_numpy(dist).to(x.dtype)
        col = torch.from_numpy(col).to(torch.long)
        row = torch.arange(col.size(0), dtype=torch.long).view(-1, 1).repeat(1, k)
        mask = ~torch.isinf(dist).view(-1)
        row, col = row.view(-1)[mask], col.view(-1)[mask]

        return torch.stack([row, col], dim=0)

