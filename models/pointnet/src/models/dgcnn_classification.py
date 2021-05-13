import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, knn, knn_graph, DynamicEdgeConv

#TODO: Remove this. Using WangYue dgcnn pytorch github
def knn(x, k):
    # print("Inside knn()")
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    # print("inner shape")
    # print(inner.shape)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    # print("xx shape")
    # print(xx.shape)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    # print("pairwise_distance shape")
    # print(pairwise_distance.shape)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    # print("Inside get_graph_feature()")
    # print("x shape")
    # print(x.shape)
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    # print("x shape after x.view")
    # print(x.shape)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
        # print("idx shape after knn")
        # print(idx.shape)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

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

class DGCNN(nn.Module):
    def __init__(self, batch_size):
        super(DGCNN, self).__init__()
        self.batch_size = batch_size
        self.k = 20
        self.edge_conv_dims = [[64, 64, 64], [64, 64, 64], [128, 128], [256, 256]]
        self.edge_convs = self.make_edge_conv_layers_()

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024 * 2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, 2)

    #def forward(self, x):
    def forward(self, data):
        # DynamicEdgeConv
        dynEdgeConv1 = list(self.edge_convs.children())[0]
        dynEdgeConv2 = list(self.edge_convs.children())[1]
        dynEdgeConv3 = list(self.edge_convs.children())[2]
        dynEdgeConv4 = list(self.edge_convs.children())[3]

        #batch_size = x.size(0)
        #x = get_graph_feature(x, k=self.k)

        x = dynEdgeConv1(data.x)
        x = self.conv1(x)
        x1 = dynEdgeConv1(x)
        #x1 = x.max(dim=-1, keepdim=False)[0]

        #x = get_graph_feature(x1, k=self.k)

        x = self.conv2(x1)
        x2 = dynEdgeConv2(x)
        #x2 = x.max(dim=-1, keepdim=False)[0]

        #x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x2)
        x3 = dynEdgeConv3(x)
        #x3 = x.max(dim=-1, keepdim=False)[0]

        #x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x3)
        x4 = dynEdgeConv4(x)
        #x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(self.batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(self.batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x

    def make_edge_conv_layers_(self):
        """Define structure of the EdgeConv Blocks
        edge_conv_dims: [[convi_mlp_dims]], e.g., [[3, 64], [64, 128]]
        """
        # print("Inside make_edge_conv_layers_()")

        layers = []
        for dims in self.edge_conv_dims:
            # mlp_dims = [dims[0] * 2] + dims[1::]
            # mlp_dims = dims
            # mlp_dims = [256, 256, 256]
            # layers.append(EdgeConv(nn=MLP(dims), aggr='max'))
            layers.append(DynamicEdgeConv(nn=MLP(dims), k=self.k, aggr='max'))
        return nn.Sequential(*layers)
