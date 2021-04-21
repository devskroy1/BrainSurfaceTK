import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import PointConv, fps, radius, global_max_pool, knn
from ..models.pointnet_randla_net import SharedMLP, LocalSpatialEncoding, AttentivePooling, LocalFeatureAggregation


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)  # TODO: FIGURE OUT THIS WITH RESPECT TO NUMBER OF POINTS
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class Net(torch.nn.Module):
    def __init__(self, num_local_features, num_global_features, num_neighbours=16, decimation=4):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decimation = decimation
        self.num_neighbours = num_neighbours
        
        super(Net, self).__init__()

        self.num_global_features = num_global_features

        # 3+6 IS 3 FOR COORDINATES, 6 FOR FEATURES PER POINT.
        # self.sa1_module = SAModule(0.5, 0.2, MLP([3 + num_local_features, 64, 64, 96]))
        # self.sa1a_module = SAModule(0.5, 0.2, MLP([96 + 3, 96, 96, 128]))
        # self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        # self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        #With randla-net modules
        self.sa1_module = SAModule(0.2, 0.2, MLP([32 + 3, 64, 64, 96]))
        self.sa1a_module = SAModule(0.5, 0.2, MLP([96 + 3, 96, 96, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.fc_start = nn.Linear(num_local_features, 8).to(self.device)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        ).to(self.device)

        # d_in = 8
        # d_out = 16

        self.mlp1 = SharedMLP(8, 8, activation_fn=nn.LeakyReLU(0.2))
        # self.mlp2 = SharedMLP(136, 32)
        self.mlp2 = SharedMLP(16, 32)
        self.shortcut = SharedMLP(8, 32, bn=True)

        self.lse1 = LocalSpatialEncoding(8, num_neighbours, self.device)
        self.lse2 = LocalSpatialEncoding(8, num_neighbours, self.device)

        self.pool1 = AttentivePooling(16, 8)
        # self.pool2 = AttentivePooling(136, 136)
        self.pool2 = AttentivePooling(16, 16)
        self.lrelu = nn.LeakyReLU()

        self.lin1 = Lin(1024 + num_global_features, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, 128)
        self.lin4 = Lin(128, 2)  # OUTPUT = NUMBER OF CLASSES, 1 IF REGRESSION TASK

    def forward(self, data):

        #First shared mlp, locSE and Attentive pooling units, then SA module
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(self.device)

        # N = input.size(1)
        N = data.size(1)
        B = data.size(0)
        d_in = data.size(2)
        d = self.decimation

        # For CUDA
        coords = data[..., :3]
        local_features = data[..., 3:]

        x = self.fc_start(local_features).transpose(-2, -1).unsqueeze(-1)
        x = self.bn_start(x)  # shape (B, d, N, 1)
        decimation_ratio = 1
        # permutation = torch.randperm(N)
        # coords = coords[:, permutation, :]
        # x = x[:, :, permutation, :]
        coords = coords[:, :N // decimation_ratio, :]

        knn_out_batch_idx = torch.zeros((B, N, self.num_neighbours), dtype=torch.int64, device=self.device)
        knn_out_batch_dist = torch.zeros((B, N, self.num_neighbours), dtype=torch.float32, device=self.device)
        for b in range(B):
            knn_coords = coords[b, :, :]
            knn_output_idx, knn_output_dist = knn(x=knn_coords, y=knn_coords, k=self.num_neighbours)
            # print("knn_output_idx shape")
            # print(knn_output_idx.shape)
            # print("knn_output_dist shape")
            # print(knn_output_dist.shape)

            knn_out_batch_idx[b] = knn_output_idx.reshape(N, self.num_neighbours)
            knn_out_batch_dist[b] = knn_output_dist.reshape(N, self.num_neighbours)
            # knn_coords = coords[b, :, :].reshape(num_points, 3)

        knn_out_batch = (knn_out_batch_idx, knn_out_batch_dist)

        features = x

        # print("features shape before self.mlp1()")
        # print(features.shape)

        x = self.mlp1(features)

        # print("x shape after self.mlp1()")
        # print(x.shape)

        x = self.lse1(coords, x, knn_out_batch)
        x = self.pool1(x)

        x = self.lse2(coords, x, knn_out_batch)
        x = self.pool2(x)

        x = self.lrelu(self.mlp2(x) + self.shortcut(features))

        d_out = x.size(1)
        coords = coords.reshape(B*N, 3)
        x = x.reshape(B*N, d_out)

        batch_cat_tensor = torch.zeros(N, dtype=torch.int64, device=self.device)
        for b in range(1, B):
            batch_tensor = torch.ones(N, dtype=torch.int64, device=self.device) * b
            # batch_tensor = batch_tensor.new_full((, npoint), b)
            # print("batch_tensor shape")
            # print(batch_tensor.shape)
            # Expected to be 512
            batch_cat_tensor = torch.cat([batch_cat_tensor, batch_tensor], dim=0)

        sa0_out = (x, coords, batch_cat_tensor)

        #sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa1a_out = self.sa1a_module(*sa1_out)
        sa2_out = self.sa2_module(*sa1a_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        # Concatenates global features to the inputs.
        if self.num_global_features > 0:
            x = torch.cat((x, data.y[:, 1:self.num_global_features + 1].view(-1, self.num_global_features)), 1)

        x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin4(x)
        return F.log_softmax(x, dim=-1)
