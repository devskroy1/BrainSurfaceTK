import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from models.pointnet.src.models.pointasnl_util import PointASNLSetAbstraction

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
    def __init__(self, num_local_features, num_global_features):
        super(Net, self).__init__()

        self.num_global_features = num_global_features

        #Start of code from pointASNL repo
        batch_size = point_cloud.get_shape()[0].value
        end_points = {}
        if use_normal:
            l0_xyz = tf.slice(point_cloud, [0, 0, 0], [-1, -1, 3])
            l0_points = tf.slice(point_cloud, [0, 0, 3], [-1, -1, 3])
        else:
            l0_xyz = point_cloud
            l0_points = point_cloud

        end_points['l0_xyz'] = l0_xyz
        as_neighbor = [12, 12] if adaptive_sample else [0, 0]

        # Set abstraction layers: pointASNL
        l1_xyz, l1_points = PointASNLSetAbstraction(l0_xyz, l0_points, npoint=512, nsample=32, mlp=[64, 64, 128],
                                                    is_training=is_training, bn_decay=bn_decay,
                                                    weight_decay=weight_decay, scope='layer1',
                                                    as_neighbor=as_neighbor[0])
        end_points['l1_xyz'] = l1_xyz
        l2_xyz, l2_points = PointASNLSetAbstraction(l1_xyz, l1_points, npoint=128, nsample=64, mlp=[128, 128, 256],
                                                    is_training=is_training, bn_decay=bn_decay,
                                                    weight_decay=weight_decay, scope='layer2',
                                                    as_neighbor=as_neighbor[1])

        #End of code from ASNL repo

        # 3+6 IS 3 FOR COORDINATES, 6 FOR FEATURES PER POINT.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3 + num_local_features, 64, 64, 96]))
        self.sa1a_module = SAModule(0.5, 0.2, MLP([96 + 3, 96, 96, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = Lin(1024 + num_global_features, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, 128)
        self.lin4 = Lin(128, 2)  # OUTPUT = NUMBER OF CLASSES, 1 IF REGRESSION TASK

    def forward(self, data):

        new_xyz, new_feature = AdaptiveSampling(grouped_xyz, new_point, as_neighbor, is_training, bn_decay,
                                                weight_decay, scope, bn)

        print("data.x shape")
        print(data.x.shape)

        print("data.pos shape")
        print(data.pos.shape)

        print("data.batch shape")
        print(data.batch.shape)

        inchannel = 6 if use_normal else 3
        pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, inchannel))
        labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
        return pointclouds_pl, labels_pl

        sa0_out = (data.x, data.pos, data.batch)
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
