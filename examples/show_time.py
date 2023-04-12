# from graph_sage_unsup_pyg import pyg_show_time
# from reddit_glzip import glzip_checktime
# from reddit_pyg import pyg_checktime

# pyg_show_time()
# from graph_sage_unsup_glzip import glzip_show_time
# glzip_show_time()

# from graph_sage_unsup_pyg import pyg_show_time
# pyg_show_time()

# pyg_checktime()
# glzip_checktime()
'''
@Author: Bohan Xu
@Date: 03/April/2023
'''

import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch_cluster import random_walk

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborSampler as RawNeighborSampler
from torch_geometric.nn import SAGEConv

# import from glzip
from glzip import CSR, GraphSageSampler
import datetime
import argparse
import pandas as pd


c = datetime.datetime.now()

EPS = 1e-15

# path = osp.join(args.root, dataset)
# dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
dt = pd.read_csv("dataset/sx-stackoverflow.txt")
data = dataset[0]

train_idx = torch.arange(data.num_nodes, dtype=torch.long)

train_loader = torch.utils.data.DataLoader(train_idx,
                                           batch_size=256,
                                           shuffle=True,
                                           drop_last=True)

csr = CSR(edge_index=data.edge_index)

glzip_sampler = GraphSageSampler(csr, [10, 10])






