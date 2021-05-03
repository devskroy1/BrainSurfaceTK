import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import PointConv, fps, radius, global_max_pool


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


class Net(torch.nn.Module):
    def __init__(self, num_local_features, num_global_features):
        super(Net, self).__init__()

        self.num_global_features = num_global_features

        # 3+num_local_features IS 3 FOR COORDINATES, num_local_features FOR FEATURES PER POINT.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3 + num_local_features, 32, 32, 64]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([64 + 3, 64, 64, 128]))
        self.sa3_module = GlobalSAModule(MLP([128 + 3, 128, 256, 512]))

        self.lin1 = Lin(512 + num_global_features, 256)
        self.lin2 = Lin(256, 128)
        self.lin3 = Lin(128, 1)

    def forward(self, data):
        # sa0_out = (data.x, data.pos, data.batch)
        # print("data.x shape")
        # print(data.x.shape)
        # print("data.pos shape")
        # print(data.pos.shape)
        decimation_ratio = 1
        d = 4
        N = data.x.size(0)
        permutation = torch.randperm(N)
        permuted_features = data.x[permutation, :]
        permuted_coords = data.pos[permutation, :]
        permuted_batch = data.batch[permutation]
        sa0_out = (permuted_features, permuted_coords, permuted_batch)

        sa1_out = self.sa1_module(*sa0_out)
        sa1x_out, sa1pos_out, sa1batch_out = sa1_out
        # print("sa1x_out shape")
        # print(sa1x_out.shape)
        # print("sa1pos_out shape")
        # print(sa1pos_out.shape)

        decimation_ratio *= d
        downsampled_sa1x_out = sa1x_out[:N//decimation_ratio, :]
        downsampled_sa1pos_out = sa1pos_out[:N // decimation_ratio, :]
        downsampled_sa1batch_out = sa1batch_out[:N // decimation_ratio]

        # print("downsampled_sa1x_out shape")
        # print(downsampled_sa1x_out.shape)
        # print("downsampled_sa1pos_out shape")
        # print(downsampled_sa1pos_out.shape)
        # print("downsampled_sa1batch_out shape")
        # print(downsampled_sa1batch_out.shape)

        sa1_out = (downsampled_sa1x_out, downsampled_sa1pos_out, downsampled_sa1batch_out)

        sa2_out = self.sa2_module(*sa1_out)
        sa2x_out, sa2pos_out, sa2batch_out = sa2_out

        decimation_ratio *= d
        downsampled_sa2x_out = sa2x_out[:N // decimation_ratio, :]
        downsampled_sa2pos_out = sa2pos_out[:N // decimation_ratio, :]
        downsampled_sa2batch_out = sa2batch_out[:N // decimation_ratio]

        sa2_out = (downsampled_sa2x_out, downsampled_sa2pos_out, downsampled_sa2batch_out)

        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        decimation_ratio *= d
        downsampled_sa3x_out = x[:N // decimation_ratio, :]

        # print("sa3_out x shape")
        # print(x.shape)
        if self.num_global_features > 0:
            downsampled_sa3x_out = torch.cat((downsampled_sa3x_out, data.y[:, 1:self.num_global_features+1].view(-1, self.num_global_features)), 1)

        x = F.relu(self.lin1(downsampled_sa3x_out))
        # print("x shape after lin1")
        # print(x.shape)
        x = F.relu(self.lin2(x))
        # print("x shape after lin2")
        # print(x.shape)
        x = self.lin3(x)
        # print("x shape after lin3")
        # print(x.shape)
        return x.view(-1)
