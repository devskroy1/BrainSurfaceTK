import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv

#TODO: Try adaptive pooling and EdgeConv for segmentn network
class DGCNNSegmentation(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, batch_size, device):
        super(DGCNNSegmentation, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, activation=nn.ReLU())
        self.conv2 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.conv3 = GraphConv(hidden_dim, n_classes, activation=None)
        self.batch_size = batch_size
        self.device = device

    def knn(self, x, k):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)

        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
        return idx

    def get_graph_feature(self, x, k=20, idx=None):
        print("Inside get_graph_feature()")
        print("x shape")
        print(x.shape)

        # print("Inside pointnet2_segmentn train()")
        # Sometimes have inconsistencies in num_points, with same batch size. With batch size 2,
        # sometimes it is 10002, sometimes 10003.
        # When 10003, it leads to error: "RuntimeError: shape '[2, 5001, 3]' is invalid for input of size 30009"
        # Have resolved this by slicing closest multiple of batch size to current num points elements from tensors
        # print("batch_tensor shape")
        # print(batch_tensor.shape)
        # print("pos_tensor.shape")
        # print(pos_tensor.shape)
        # print("x_tensor shape")
        # print(x_tensor.shape)
        # print("y_tensor shape")
        # print(y_tensor.shape)

        tot_num_points = x.size(0)
        num_features = x.size(1)

        # print("batch size")
        # print(batch_size)

        quot = tot_num_points // self.batch_size
        num_points_multiple = quot * self.batch_size

        x = x[:num_points_multiple, :]



        x = x.reshape(self.batch_size, quot, num_features)
        #batch_size = x.size(0)
        num_points = x.size(1)
        x = x.view(self.batch_size, num_features, num_points)
        if idx is None:
            idx = self.knn(x, k=k)  # (batch_size, num_points, k)

        idx_base = torch.arange(0, self.batch_size, device=self.device).view(-1, 1, 1) * num_points

        idx = idx + idx_base

        idx = idx.view(-1)

        _, num_dims, _ = x.size()

        x = x.transpose(2,
                        1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = x.view(self.batch_size * num_points, -1)[idx, :]
        feature = feature.view(self.batch_size, num_points, k, num_dims)
        x = x.view(self.batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

        return feature

    def forward(self, graph, features):
        # Perform graph convolution and activation function.
        print("features shape before calling get_graph_feature()")
        print(features.shape)
        features = self.get_graph_feature(features, k=20)
        print("graph")
        print(graph)
        print("features shape from first get_graph_feature()")
        print(features.shape)

        hidden = self.conv1(graph, features)
        hidden = self.get_graph_feature(hidden, k=20)
        hidden = self.conv2(graph, hidden)
        hidden = self.get_graph_feature(hidden, k=20)
        return self.conv3(graph, hidden)

