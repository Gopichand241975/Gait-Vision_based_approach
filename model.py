"""ResGCN-style gait feature extractor on the 17-joint skeleton graph."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


def build_adjacency(edges, num_nodes):
    a = np.eye(num_nodes, dtype=np.float32)
    for i, j in edges:
        a[i, j] = a[j, i] = 1.0
    d = a.sum(1)
    return torch.from_numpy(a / np.sqrt(d[:, None] * d[None, :]))


class GCNBlock(nn.Module):
    """Spatial graph convolution + temporal convolution with a residual connection."""

    def __init__(self, cin, cout, adj, stride=1, dropout=0.1, residual=True):
        super().__init__()
        self.register_buffer("adj", adj)
        self.edge_weight = nn.Parameter(torch.ones_like(adj))
        self.spatial = nn.Conv2d(cin, cout, 1)
        self.bn1 = nn.BatchNorm2d(cout)
        self.temporal = nn.Sequential(
            nn.Conv2d(cout, cout, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(cout), nn.Dropout(dropout))
        if not residual:
            self.res = None
        elif cin == cout and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout))

    def forward(self, x):                       # x: (N, C, T, V)
        res = 0 if self.res is None else self.res(x)
        x = self.spatial(x)
        x = torch.einsum("nctv,vw->nctw", x, self.adj * self.edge_weight)
        x = F.relu(self.bn1(x))
        return F.relu(self.temporal(x) + res)


class GaitResGCN(nn.Module):
    """Input (N, 7, T, 17) -> L2-normalised gait feature (N, FEATURE_DIM)."""

    def __init__(self, num_classes=0, in_channels=config.IN_CHANNELS,
                 feature_dim=config.FEATURE_DIM, dropout=config.DROPOUT):
        super().__init__()
        v = config.NUM_JOINTS
        adj = build_adjacency(config.SKELETON_EDGES, v)
        self.data_bn = nn.BatchNorm1d(in_channels * v)
        spec = [(in_channels, 32, 1), (32, 32, 1), (32, 64, 2),
                (64, 64, 1), (64, 128, 2), (128, 128, 1)]
        self.blocks = nn.ModuleList(
            GCNBlock(ci, co, adj, s, dropout, residual=(i > 0)) for i, (ci, co, s) in enumerate(spec))
        self.fc = nn.Linear(128, feature_dim)
        self.num_classes = num_classes
        if num_classes > 0:
            self.class_weight = nn.Parameter(torch.randn(num_classes, feature_dim) * 0.01)

    def forward(self, x):
        n, c, t, v = x.shape
        x = self.data_bn(x.permute(0, 1, 3, 2).reshape(n, c * v, t))
        x = x.reshape(n, c, v, t).permute(0, 1, 3, 2)
        for b in self.blocks:
            x = b(x)
        return F.normalize(self.fc(x.mean(dim=(2, 3))), dim=1)

    def classify(self, feat):
        """Cosine classifier over training identities (training only)."""
        return config.COSINE_SCALE * F.linear(feat, F.normalize(self.class_weight, dim=1))