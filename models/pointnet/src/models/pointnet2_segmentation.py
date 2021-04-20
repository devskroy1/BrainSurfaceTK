import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from torch_geometric.nn import knn, knn_interpolate
from ..models.pointnet_randla_net import SharedMLP, LocalSpatialEncoding, AttentivePooling, LocalFeatureAggregation

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        #print("Inside SAModule forward")
        # print("x shape before calling PointConv")
        # print(x.shape)
        idx = fps(pos, batch, ratio=self.ratio)

        # print("pos shape")
        # print(pos.shape)
        # print("pos[idx] shape")
        # print(pos[idx].shape)
        #
        # print("batch shape")
        # print(batch.shape)
        # print("batch[idx] shape")
        # print(batch[idx].shape)

        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)

        # edge_index = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)
        # print("edge_index shape from radius")
        # print(edge_index.shape)

        # print("x shape")
        # print(x.shape)
        # print("pos shape")
        # print(pos.shape)
        # print("idx shape")
        # print(idx.shape)
        # print("pos[idx] shape")
        # print(pos[idx].shape)

        x = self.conv(x, (pos, pos[idx]), edge_index)
        # print("x shape after calling PointConv")
        # print(x.shape)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        # print("Inside GlobalSAModule forward()")

        x = self.nn(torch.cat([x, pos], dim=1))
        # print("x shape after MLP")
        # print(x.shape)
        x = global_max_pool(x, batch)
        # print("x shape after global_max_pool()")
        # print(x.shape)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        # print("Inside FPModule forward")
        # print("x shape")
        # print(x.shape)
        # print("x_skip shape")
        # print(x_skip.shape)
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        # print("x shape before calling self.nn")
        # print(x.shape)
        x = self.nn(x)
        # print("x shape after calling self.nn")
        # print(x.shape)
        return x, pos_skip, batch_skip


# My network
class Net(torch.nn.Module):
    def __init__(self, num_classes, num_local_features, num_neighbours=16, decimation=4, num_global_features=None):
        #TODO: SEE HOW YOU CAN USE GLOBAL FEATURES
        '''
        :param num_classes: Number of segmentation classes
        :param num_local_features: Feature per node
        :param num_global_features: NOT USED
        '''

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decimation = decimation
        super(Net, self).__init__()
        #self.sa1_module = SAModule(0.2, 0.2, MLP([3 + num_local_features, 64, 64, 128]))
        # self.sa1_module = SAModule(0.2, 0.2, MLP([8 + num_local_features, 64, 64, 152]))
        # self.sa2_module = SAModule(0.25, 0.4, MLP([152 + 3, 128, 128, 256]))

        self.sa1_module = SAModule(0.2, 0.2, MLP([8 + num_local_features, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([136 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 136, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + 8, 128, 128, 128]))
        #self.fp1_module = FPModule(3, MLP([128 + num_local_features, 128, 128, 128]))

        self.fc_start = nn.Linear(num_local_features, 8).to(self.device)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        ).to(self.device)

        # d_in = 8
        # d_out = 16

        self.mlp1 = SharedMLP(8, 8, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(136, 32)
        self.shortcut = SharedMLP(8, 32, bn=True)

        self.lse1 = LocalSpatialEncoding(8, num_neighbours, self.device)
        self.lse2 = LocalSpatialEncoding(8, num_neighbours, self.device)

        self.pool1 = AttentivePooling(16, 8)
        self.pool2 = AttentivePooling(136, 136)

        # self.encoder = nn.ModuleList([
        #     LocalFeatureAggregation(8, 16, num_neighbours, self.device),
        #     LocalFeatureAggregation(32, 64, num_neighbours, self.device),
        #     LocalFeatureAggregation(128, 128, num_neighbours, self.device),
        #     LocalFeatureAggregation(256, 256, num_neighbours, self.device)
        # ])
        #
        # decoder_kwargs = dict(
        #     transpose=True,
        #     bn=True,
        #     activation_fn=nn.ReLU()
        # )
        #
        # self.decoder = nn.ModuleList([
        #     SharedMLP(1024, 512, **decoder_kwargs),
        #     SharedMLP(512, 256, **decoder_kwargs),
        #     SharedMLP(256, 128, **decoder_kwargs),
        #     SharedMLP(128, 128, **decoder_kwargs)
        # ])

        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, num_classes)
        self.num_neighbours = num_neighbours

    # def forward(self, data):
    #
    #     print("Inside PointNet segmentation forward()")
    #     N = data.x.size(0)
    #     d = self.decimation
    #
    #     sa0_out = (data.x, data.pos, data.batch)
    #     sa1_out = self.sa1_module(*sa0_out)
    #     print("Just before calling sa2 module forward")
    #     sa2_out = self.sa2_module(*sa1_out)
    #     #sa3_out = self.sa3_module(*sa2_out)
    #     x, coords, batch = self.sa3_module(*sa2_out)
    #
    #     # fp3_out = self.fp3_module(*sa3_out, *sa2_out)
    #     # fp2_out = self.fp2_module(*fp3_out, *sa1_out)
    #     # x, coords, batch = self.fp1_module(*fp2_out, *sa0_out)
    #
    #     print("x shape from sa3 module")
    #     print(x.shape)
    #     print("coords shape from sa3 module")
    #     print("coords shape")
    #     print(coords.shape)
    #
    #     decimation_ratio = 1
    #     # coords = data.pos.clone().cpu()
    #     # x = data.x.clone().cpu()
    #     # print("coords")
    #     # print(coords)
    #     # permutation = torch.randperm(N)
    #     # coords = coords[permutation]
    #     # x = x[permutation]
    #     #
    #     # print("x.shape")
    #     # print(x.shape)
    #     # print("x")
    #     # print(x)
    #
    #     x = x.view(x.size(0), x.size(1)//16, 16, 1)
    #     print("x.shape after reshaping into 4d")
    #     print(x.shape)
    #
    #     for lfa in self.encoder:
    #         # at iteration i, x.shape = (B, N//(d**i), d_in)
    #
    #         x = lfa(coords[:, :N//decimation_ratio], x)
    #         x_stack.append(x.clone())
    #         decimation_ratio *= d
    #         x = x[:, :, :N//decimation_ratio]
    #
    #     for mlp in self.decoder:
    #         neighbors, _ = knn(
    #             coords[:, :N//decimation_ratio].cpu().contiguous(), # original set
    #             coords[:, :d*N//decimation_ratio].cpu().contiguous(), # upsampled set
    #             1
    #         ) # shape (B, N, 1)
    #         neighbors = neighbors.to(self.device)
    #
    #         extended_neighbors = neighbors.unsqueeze(1).expand(-1, x.size(1), -1, 1)
    #
    #         x_neighbors = torch.gather(x, -2, extended_neighbors)
    #
    #         x = torch.cat((x_neighbors, x_stack.pop()), dim=1)
    #
    #         x = mlp(x)
    #
    #         decimation_ratio //= d
    #
    #     x = x[:, :, torch.argsort(permutation)]
    #
    #     x = F.relu(self.lin1(x))
    #     x = F.dropout(x, p=0.5, training=self.training)
    #     x = self.lin2(x)
    #     x = F.dropout(x, p=0.5, training=self.training)
    #     x = self.lin3(x)
    #     return F.log_softmax(x, dim=-1)

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

        # print("x shape")
        # print(x.shape)
        # print("coords shape")
        # print(coords.shape)
        # print("batch_cat_tensor shape")
        # print(batch_cat_tensor.shape)

        #sa0_out = (x.to(self.device), coords.to(self.device), batch_cat_tensor.to(self.device))
        sa0_out = (x, coords, batch_cat_tensor)

        #sa1_out_x, sa1_out_pos, sa1_out_batch = self.sa1_module(*sa0_out)
        sa1_out = self.sa1_module(*sa0_out)

        # print("sa1_out_x shape")
        # print(sa1_out_x.shape)
        # print("sa1_out_pos shape")
        # print(sa1_out_pos.shape)
        # print("sa1_out_batch shape")
        # print(sa1_out_batch.shape)

        #Randla-net code
        # N = sa1_out_x.size(0) // B
        # num_features = sa1_out_x.size(1)
        # sa1_out_x = sa1_out_x.reshape(B, num_features, N).unsqueeze(-1)
        # sa1_out_pos = sa1_out_pos.reshape(B, N, 3)
        # knn_out_batch_idx = torch.zeros((B, N, self.num_neighbours), dtype=torch.int64, device=self.device)
        # knn_out_batch_dist = torch.zeros((B, N, self.num_neighbours), dtype=torch.float32, device=self.device)
        # for b in range(B):
        #     knn_coords = sa1_out_pos[b, :, :]
        #     knn_output_idx, knn_output_dist = knn(x=knn_coords, y=knn_coords, k=self.num_neighbours)
        #     # print("knn_output_idx shape")
        #     # print(knn_output_idx.shape)
        #     # print("knn_output_dist shape")
        #     # print(knn_output_dist.shape)
        #
        #     knn_out_batch_idx[b] = knn_output_idx.reshape(N, self.num_neighbours)
        #     knn_out_batch_dist[b] = knn_output_dist.reshape(N, self.num_neighbours)
        #     # knn_coords = coords[b, :, :].reshape(num_points, 3)
        #
        # knn_out_batch = (knn_out_batch_idx, knn_out_batch_dist)
        #
        # x = self.lse2(sa1_out_pos, sa1_out_x, knn_out_batch)
        # # print("x shape after self.lse2()")
        # # print(x.shape)
        # x = self.pool2(x)
        # # print("x shape after pool2")
        # # print(x.shape)
        #
        # # print("x shape before mlp2")
        # # print(x.shape)
        # # print("features shape")
        # # print(features.shape)
        # # mlp2 = self.mlp2(x)
        # # shortcut = self.shortcut(features)
        # #
        # # npoint = shortcut.size(2)
        # # nchannels = shortcut.size(1)
        # # print("npoint")
        # # print(npoint)
        # # #Should be 5001
        # # print("mlp2 shape")
        # # print(mlp2.shape)
        # # print("shortcut shape")
        # # print(shortcut.shape)
        # # mlp2 = mlp2.expand(B, nchannels, npoint, 1)
        # # x = nn.LeakyReLU(mlp2 + shortcut)
        # #
        # # print("x shape after addition of mlp2 and shortcut connection")
        # # print(x.shape)
        #
        # d_out = x.size(1)
        # coords = sa1_out_pos.reshape(B*N, 3)
        # x = x.reshape(B*N, d_out)
        #
        # batch_cat_tensor = torch.zeros(N, dtype=torch.int64, device=self.device)
        # for b in range(1, B):
        #     batch_tensor = torch.ones(N, dtype=torch.int64, device=self.device) * b
        #     # batch_tensor = batch_tensor.new_full((, npoint), b)
        #     # print("batch_tensor shape")
        #     # print(batch_tensor.shape)
        #     # Expected to be 512
        #     batch_cat_tensor = torch.cat([batch_cat_tensor, batch_tensor], dim=0)

        #sa1_out = (x, coords, batch_cat_tensor)

        #sa2_out_x, sa2_out_pos, sa2_out_batch = self.sa2_module(*sa1_out)
        #End of Randla-net code

        #Vanilla Pointnet++
        sa2_out = self.sa2_module(*sa1_out)
        #sa2_out_x, sa2_out_pos, sa2_out_batch = sa2_out

        #
        # print("sa2_out_x shape")
        # print(sa2_out_x.shape)
        # print("sa2_out_pos shape")
        # print(sa2_out_pos.shape)
        # print("sa2_out_batch shape")
        # print(sa2_out_batch.shape)

        sa3_out = self.sa3_module(*sa2_out)
        #sa3_out_x, sa3_out_pos, sa3_out_batch = sa3_out

        # print("sa3_out_x shape")
        # print(sa3_out_x.shape)
        # print("sa3_out_pos shape")
        # print(sa3_out_pos.shape)
        # print("sa3_out_batch shape")
        # print(sa3_out_batch.shape)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)

