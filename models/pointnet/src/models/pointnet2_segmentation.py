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
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
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


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


# My network
class Net(torch.nn.Module):
    def __init__(self, num_classes, num_local_features, num_neighbours=16, decimation=4, num_global_features=None):
        '''
        :param num_classes: Number of segmentation classes
        :param num_local_features: Feature per node
        :param num_global_features: NOT USED
        '''

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.decimation = decimation
        super(Net, self).__init__()
        self.sa1_module = SAModule(0.2, 0.2, MLP([3 + num_local_features, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + num_local_features, 128, 128, 128]))

        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(8, 16, num_neighbours, self.device),
            LocalFeatureAggregation(32, 64, num_neighbours, self.device),
            LocalFeatureAggregation(128, 128, num_neighbours, self.device),
            LocalFeatureAggregation(256, 256, num_neighbours, self.device)
        ])

        decoder_kwargs = dict(
            transpose=True,
            bn=True,
            activation_fn=nn.ReLU()
        )

        self.decoder = nn.ModuleList([
            SharedMLP(1024, 512, **decoder_kwargs),
            SharedMLP(512, 256, **decoder_kwargs),
            SharedMLP(256, 128, **decoder_kwargs),
            SharedMLP(128, 128, **decoder_kwargs)
        ])

        self.lin1 = torch.nn.Linear(128, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, num_classes)

    def forward(self, data):
        N = data.x.size(0)
        d = self.decimation

        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        decimation_ratio = 1
        coords = data.pos.clone().cpu()
        x = data.x.clone().cpu()
        print("coords")
        print(coords)
        permutation = torch.randperm(N)
        coords = coords[permutation]
        x = x[permutation]

        for lfa in self.encoder:
            # at iteration i, x.shape = (B, N//(d**i), d_in)
            x = lfa(coords[:N//decimation_ratio], x)
            x_stack.append(x.clone())
            decimation_ratio *= d
            x = x[:N//decimation_ratio]

        for mlp in self.decoder:
            neighbors, _ = knn(
                coords[:, :N//decimation_ratio].cpu().contiguous(), # original set
                coords[:, :d*N//decimation_ratio].cpu().contiguous(), # upsampled set
                1
            ) # shape (B, N, 1)
            neighbors = neighbors.to(self.device)

            extended_neighbors = neighbors.unsqueeze(1).expand(-1, x.size(1), -1, 1)

            x_neighbors = torch.gather(x, -2, extended_neighbors)

            x = torch.cat((x_neighbors, x_stack.pop()), dim=1)

            x = mlp(x)

            decimation_ratio //= d

        x = x[:, :, torch.argsort(permutation)]

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)