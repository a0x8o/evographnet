# Databricks notebook source
# MAGIC %md ## EvoGraphNet on Ray-on-Databricks
# MAGIC
# MAGIC #### @author stephen.offer@databricks.com
# MAGIC
# MAGIC #### date 02-Feb-2024

# COMMAND ----------

# MAGIC %pip install "ray[all]@https://github.com/WeichenXu123/packages/raw/c5d6cedacec0ec2446a8c0803b14f35937b5fe0e/ray/spark-df-loader/ray-3.0.0.dev0-cp310-cp310-linux_x86_64.whl" 

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster
import ray

from ray.runtime_env import RuntimeEnv

restart = True

if restart is True:
     if ray.is_initialized():
          shutdown_ray_cluster()
          ray.shutdown()

if not ray.is_initialized():
     setup_ray_cluster(
          max_worker_nodes=3,
          min_worker_nodes=3,
          num_cpus_per_node=8,
          num_gpus_per_node=1,
          num_gpus_head_node=1,
          num_cpus_head_node=8,
          collect_log_to_path="/dbfs/soffer/ray_logs"
        )
     runtime_env = {"env_vars": {"GLOO_SOCKET_IFNAME":"eth0"}}
     ray.init(ignore_reinit_error=True, runtime_env=RuntimeEnv(env_vars = runtime_env['env_vars']))

print(ray.cluster_resources())

# COMMAND ----------

import argparse
import os
import os.path as osp
import numpy as np
import math
import itertools
import copy
import pickle
from sys import exit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Tanh, Dropout, LeakyReLU
from torch.autograd import Variable
from torch.distributions import normal, kl

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import NNConv, BatchNorm, EdgePooling, TopKPooling, global_add_pool
from torch_geometric.utils import get_laplacian, to_dense_adj

import matplotlib.pyplot as plt

import os
import os.path as osp
import numpy as np
import math
import itertools
import copy
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Tanh, Dropout, LeakyReLU
from torch.autograd import Variable
from torch.distributions import normal

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import NNConv, BatchNorm, EdgePooling, TopKPooling, global_add_pool
from torch_geometric.utils import get_laplacian, to_dense_adj

import matplotlib.pyplot as plt


class MRDataset(InMemoryDataset):

    def __init__(self, root, src, dest, h, connectomes=1, subs=1000, transform=None, pre_transform=None):

        """
        src: Input to the model
        dest: Target output of the model
        h: Load LH or RH data
        subs: Maximum number of subjects

        Note: Since we do not reprocess the data if it is already processed, processed files should be
        deleted if there is any change in the data we are reading.
        """
        self.root = root
        self.src, self.dest, self.h, self.subs, self.connectomes = src, dest, h, subs, connectomes
        super(MRDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def data_read(self, h="lh", nbr_of_subs=1000, connectomes=1):

        """
        Takes the (maximum) number of subjects and hemisphere we are working on
        as arguments, returns t0, t1, t2's of the connectomes for each subject
        in a single torch.FloatTensor.
        """

        subs = None  # Subjects

        data_path = self.root

        for i in range(1, nbr_of_subs):
            s = data_path + "/cortical." + h.lower() + ".ShapeConnectivityTensor_OAS2_"
            if i < 10:
                s += "0"
            s += "00" + str(i) + "_"

            for mr in ["MR1", "MR2"]:
                try:  # Sometimes subject we are looking for does not exist
                    t0 = np.loadtxt(s + mr + "_t0.txt")
                    t1 = np.loadtxt(s + mr + "_t1.txt")
                    t2 = np.loadtxt(s + mr + "_t2.txt")
                except:
                    continue

                # Read the connectomes at t0, t1 and t2, then stack them
                read_limit = (connectomes * 35)
                t_stacked = np.vstack((t0[:read_limit, :], t1[:read_limit, :], t2[:read_limit, :]))
                tsr = t_stacked.reshape(3, connectomes * 35, 35)

                if subs is None:  # If first subject
                    subs = tsr
                else:
                    subs = np.vstack((subs, tsr))

        # Then, reshape to match the shape of the model's expected input shape
        # final_views should be a torch tensor or Pytorch Geometric complains
        final_views = torch.tensor(np.moveaxis(subs.reshape(-1, 3, (connectomes * 35), 35), 1, -1), dtype=torch.float)

        return final_views

    @property
    def processed_file_names(self):
        return [
            "data_" + str(self.connectomes) + "_" + self.h.lower() + "_" + str(self.subs) + "_" + str(self.src) + str(
                self.dest) + ".pt"]

    def process(self):

        """
        Prepares the data for PyTorch Geometric.
        """

        unprocessed = self.data_read(self.h, self.subs)
        num_samples, timestamps = unprocessed.shape[0], unprocessed.shape[-1]
        assert 0 <= self.dest <= timestamps
        assert 0 <= self.src <= timestamps

        # Turn the data into PyTorch Geometric Graphs
        data_list = list()

        for sample in range(num_samples):
            x = unprocessed[sample, :, :, self.src]
            y = unprocessed[sample, :, :, self.dest]

            edge_index, edge_attr, rows, cols = create_edge_index_attribute(x)
            y_edge_index, y_edge_attr, _, _ = create_edge_index_attribute(y)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                        y=y, y_edge_index=y_edge_index, y_edge_attr=y_edge_attr)

            data.num_nodes = rows
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MRDataset2(InMemoryDataset):

    def __init__(self, root, h, connectomes=1, subs=1000, transform=None, pre_transform=None):

        """
        src: Input to the model
        dest: Target output of the model
        h: Load LH or RH data
        subs: Maximum number of subjects

        Note: Since we do not reprocess the data if it is already processed, processed files should be
        deleted if there is any change in the data we are reading.
        """
        self.root = root
        self.h, self.subs, self.connectomes = h, subs, connectomes
        super(MRDataset2, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def data_read(self, h="lh", nbr_of_subs=1000, connectomes=1):

        """
        Takes the (maximum) number of subjects and hemisphere we are working on
        as arguments, returns t0, t1, t2's of the connectomes for each subject
        in a single torch.FloatTensor.
        """

        subs = None  # Subjects

        data_path = self.root

        for i in range(1, nbr_of_subs):
            s = data_path + "/cortical." + h.lower() + ".ShapeConnectivityTensor_OAS2_"
            if i < 10:
                s += "0"
            s += "00" + str(i) + "_"

            for mr in ["MR1", "MR2"]:
                try:  # Sometimes subject we are looking for does not exist
                    t0 = np.loadtxt(s + mr + "_t0.txt")
                    t1 = np.loadtxt(s + mr + "_t1.txt")
                    t2 = np.loadtxt(s + mr + "_t2.txt")
                except:
                    continue

                # Read the connectomes at t0, t1 and t2, then stack them
                read_limit = (connectomes * 35)
                t_stacked = np.vstack((t0[:read_limit, :], t1[:read_limit, :], t2[:read_limit, :]))
                tsr = t_stacked.reshape(3, connectomes * 35, 35)

                if subs is None:  # If first subject
                    subs = tsr
                else:
                    subs = np.vstack((subs, tsr))

        # Then, reshape to match the shape of the model's expected input shape
        # final_views should be a torch tensor or Pytorch Geometric complains
        final_views = torch.tensor(np.moveaxis(subs.reshape(-1, 3, (connectomes * 35), 35), 1, -1), dtype=torch.float)

        return final_views

    @property
    def processed_file_names(self):
        return [
            "2data_" + str(self.connectomes) + "_" + self.h.lower() + "_" + str(self.subs) + "_" + ".pt"]

    def process(self):

        """
        Prepares the data for PyTorch Geometric.
        """

        unprocessed = self.data_read(self.h, self.subs)
        num_samples, timestamps = unprocessed.shape[0], unprocessed.shape[-1]

        # Turn the data into PyTorch Geometric Graphs
        data_list = list()

        for sample in range(num_samples):
            x = unprocessed[sample, :, :, 0]
            y = unprocessed[sample, :, :, 1]
            y2 = unprocessed[sample, :, :, 2]

            edge_index, edge_attr, rows, cols = create_edge_index_attribute(x)
            y_edge_index, y_edge_attr, _, _ = create_edge_index_attribute(y)
            y2_edge_index, y2_edge_attr, _, _ = create_edge_index_attribute(y2)
            y_distr = normal.Normal(y.mean(dim=1), y.std(dim=1))
            y2_distr = normal.Normal(y2.mean(dim=1), y2.std(dim=1))
            y_lap_ei, y_lap_ea = get_laplacian(y_edge_index, y_edge_attr)
            y2_lap_ei, y2_lap_ea = get_laplacian(y2_edge_index, y2_edge_attr)
            y_lap = to_dense_adj(y_lap_ei, edge_attr=y_lap_ea)
            y2_lap = to_dense_adj(y2_lap_ei, edge_attr=y2_lap_ea)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                        y=y, y_edge_index=y_edge_index, y_edge_attr=y_edge_attr, y_distr=y_distr,
                        y2=y2, y2_edge_index=y2_edge_index, y2_edge_attr=y2_edge_attr, y2_distr=y2_distr,
                        y_lap=y_lap, y2_lap=y2_lap)

            data.num_nodes = rows
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def create_edge_index_attribute(adj_matrix):
    """
    Given an adjacency matrix, this function creates the edge index and edge attribute matrix
    suitable to graph representation in PyTorch Geometric.
    """

    rows, cols = adj_matrix.shape[0], adj_matrix.shape[1]
    edge_index = torch.zeros((2, rows * cols), dtype=torch.long)
    edge_attr = torch.zeros((rows * cols, 1), dtype=torch.float)
    counter = 0

    for src, attrs in enumerate(adj_matrix):
        for dest, attr in enumerate(attrs):
            edge_index[0][counter], edge_index[1][counter] = src, dest
            edge_attr[counter] = attr
            counter += 1

    return edge_index, edge_attr, rows, cols


def swap(data):
    # Swaps the x & y values of the given graph
    edge_i, edge_attr, _, _ = create_edge_index_attribute(data.y)
    data_s = Data(x=data.y, edge_index=edge_i, edge_attr=edge_attr, y=data.x)
    return data_s


def cross_val_indices(folds, num_samples, root="../data/", new=False):
    """
    Takes the number of inputs and number of folds.
    Determines indices to go into validation split in each turn.
    Saves the indices on a file for experimental reproducibility and does not overwrite
    the already determined indices unless new=True.
    """

    kf = KFold(n_splits=folds, shuffle=True)
    train_indices = list()
    val_indices = list()

    try:
        if new == True:
            raise IOError
        with open(root + str(folds) + "_" + str(num_samples) + "cv_train", "rb") as f:
            train_indices = pickle.load(f)
        with open(root+ str(folds) + "_" + str(num_samples) + "cv_val", "rb") as f:
            val_indices = pickle.load(f)
    except IOError:
        for tr_index, val_index in kf.split(np.zeros((num_samples, 1))):
            train_indices.append(tr_index)
            val_indices.append(val_index)
        with open(root + str(folds) + "_" + str(num_samples) + "cv_train", "wb") as f:
            pickle.dump(train_indices, f)
        with open(root + str(folds) + "_" + str(num_samples) + "cv_val", "wb") as f:
            pickle.dump(val_indices, f)

    return train_indices, val_indices
  
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Tanh, Dropout, LeakyReLU
from torch.autograd import Variable

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import NNConv, BatchNorm, EdgePooling, TopKPooling, global_add_pool

import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        lin = Sequential(Linear(1, 1225), ReLU())
        self.conv1 = NNConv(35, 35, lin, aggr='mean', root_weight=True, bias=True)
        self.conv11 = BatchNorm(35, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        lin = Sequential(Linear(1, 35), ReLU())
        self.conv2 = NNConv(35, 1, lin, aggr='mean', root_weight=True, bias=True)
        self.conv22 = BatchNorm(1, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        lin = Sequential(Linear(1, 35), ReLU())
        self.conv3 = NNConv(1, 35, lin, aggr='mean', root_weight=True, bias=True)
        self.conv33 = BatchNorm(35, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x1 = torch.sigmoid(self.conv11(self.conv1(x, edge_index, edge_attr)))
        x1 = F.dropout(x1, training=self.training)
        #Below 2 lines are the corrections
        x1 = (x1 + x1.T) / 2.0
        x1.fill_diagonal_(fill_value = 0)
        x2 = torch.sigmoid(self.conv22(self.conv2(x1, edge_index, edge_attr)))
        x2 = F.dropout(x2, training=self.training)

        x3 = torch.cat([torch.sigmoid(self.conv33(self.conv3(x2, edge_index, edge_attr))), x1], dim=1)
        x4 = x3[:, 0:35]
        x5 = x3[:, 35:70]

        x6 = (x4 + x5) / 2
        #Below 2 lines are the corrections
        x6 = (x6 + x6.T) / 2.0
        x6.fill_diagonal_(fill_value = 0)
        return x6


class Discriminator(torch.nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        lin = Sequential(Linear(2, 1225), ReLU())
        self.conv1 = NNConv(35, 35, lin, aggr='mean', root_weight=True, bias=True)
        self.conv11 = BatchNorm(35, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

        lin = Sequential(Linear(2, 35), ReLU())
        self.conv2 = NNConv(35, 1, lin, aggr='mean', root_weight=True, bias=True)
        self.conv22 = BatchNorm(1, eps=1e-03, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, data, data_to_translate):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_attr_data_to_translate = data_to_translate.edge_attr

        edge_attr_data_to_translate_reshaped = edge_attr_data_to_translate.view(1225, 1)

        gen_input = torch.cat((edge_attr, edge_attr_data_to_translate_reshaped), -1)
        x = F.relu(self.conv11(self.conv1(x, edge_index, gen_input)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv22(self.conv2(x, edge_index, gen_input)))

        return torch.sigmoid(x)


torch.manual_seed(0)  # To get the same results across experiments

# if torch.cuda.is_available():
#     device = torch.device('cuda')
#     print('running on GPU')
# else:
#     device = torch.device("cpu")
#     print('running on CPU')

# Parser
# parser = argparse.ArgumentParser()
# parser.add_argument('--lr_g', type=float, default=0.01, help='Generator learning rate')
# parser.add_argument('--lr_d', type=float, default=0.0002, help='Discriminator learning rate')
# parser.add_argument('--loss', type=str, default='BCE', help='Which loss to use for training',
#                     choices=['BCE', 'LS'])
# parser.add_argument('--batch', type=int, default=1, help='Batch Size')
# parser.add_argument('--epoch', type=int, default=500, help='How many epochs to train')
# parser.add_argument('--folds', type=int, default=3, help='How many folds for CV')
# parser.add_argument('--tr_st', type=str, default='same', help='Training strategy',
#                     choices=['same', 'turns', 'idle'])
# parser.add_argument('--id_e', type=int, default=2, help='If training strategy is idle, for how many epochs')
# parser.add_argument('--exp', type=int, default=0, help='Which experiment are you running')
# parser.add_argument('--tp_c', type=float, default=0.0, help='Coefficient of topology loss')
# parser.add_argument('--g_c', type=float, default=2.0, help='Coefficient of adversarial loss')
# parser.add_argument('--i_c', type=float, default=2.0, help='Coefficient of identity loss')
# parser.add_argument('--kl_c', type=float, default=0.001, help='Coefficient of KL loss')
# parser.add_argument('--decay', type=float, default=0.0, help='Weight Decay')
# opt = parser.parse_args()

class Opt(object):

  def __init__(self):
    self.lr_g = 0.01
    self.lr_d = 0.0002
    self.loss = 'BCE'
    self.batch = 1
    self.epoch = 500
    self.folds = 16
    self.tr_st = 'same'
    self.id_e = 2
    self.exp = 0
    self.tp_c = 0.0
    self.g_c = 2.0
    self.i_c = 2.0
    self.kl_c = 0.001
    self.decay = 0.0

opt = Opt()

# Datasets

data_root = "/dbfs/soffer/evograph/"

h_data = MRDataset2(data_root, "lh", subs=989)

# Parameters

batch_size = opt.batch
lr_G = opt.lr_g
lr_D = opt.lr_d
num_epochs = opt.epoch
folds = opt.folds

connectomes = 1
train_generator = 1

# Coefficients for loss
i_coeff = opt.i_c
g_coeff = opt.g_c
kl_coeff = opt.kl_c
tp_coeff = opt.tp_c

if opt.tr_st != 'idle':
    opt.id_e = 0

# Training

train_ind, val_ind = cross_val_indices(folds, len(h_data), root=data_root)

# Saving the losses for the future
gen_mae_losses_tr = None
disc_real_losses_tr = None
disc_fake_losses_tr = None
gen_mae_losses_val = None
disc_real_losses_val = None
disc_fake_losses_val = None
gen_mae_losses_tr2 = None
disc_real_losses_tr2 = None
disc_fake_losses_tr2 = None
gen_mae_losses_val2 = None
disc_real_losses_val2 = None
disc_fake_losses_val2 = None
k1_train_s = None
k2_train_s = None
k1_val_s = None
k2_val_s = None
tp1_train_s = None
tp2_train_s = None
tp1_val_s = None
tp2_val_s = None
gan1_train_s = None
gan2_train_s = None
gan1_val_s = None
gan2_val_s = None


@ray.remote(num_cpus=2, num_gpus=0.25)
def run_fold(fold):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('running on GPU')
    else:
        device = torch.device("cpu")
        print('running on CPU')
    loss_dict = {"BCE": torch.nn.BCELoss().to(device),
             "LS": torch.nn.MSELoss().to(device)}

    adversarial_loss = loss_dict[opt.loss.upper()]
    identity_loss = torch.nn.L1Loss().to(device)  # Will be used in training
    msel = torch.nn.MSELoss().to(device)
    mael = torch.nn.L1Loss().to(device)  # Not to be used in training (Measure generator success)
    counter_g, counter_d = 0, 0
    tp = torch.nn.MSELoss().to(device)  # Used for node strength

    train_set, val_set = h_data[list(train_ind[fold])], h_data[list(val_ind[fold])]
    h_data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    h_data_test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    val_step = len(h_data_test_loader)

    for data in h_data_train_loader:  # Determine the maximum number of samples in a batch
        data_size = data.x.size(0)
        break

    # Create generators and discriminators
    generator = Generator().to(device)
    generator2 = Generator().to(device)
    discriminator = Discriminator().to(device)
    discriminator2 = Discriminator().to(device)

    optimizer_G = torch.optim.AdamW(generator.parameters(), lr=lr_G, betas=(0.5, 0.999), weight_decay=opt.decay)
    optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999), weight_decay=opt.decay)
    optimizer_G2 = torch.optim.AdamW(generator2.parameters(), lr=lr_G, betas=(0.5, 0.999), weight_decay=opt.decay)
    optimizer_D2 = torch.optim.AdamW(discriminator2.parameters(), lr=lr_D, betas=(0.5, 0.999), weight_decay=opt.decay)

    total_step = len(h_data_train_loader)
    real_label = torch.ones((data_size, 1)).to(device)
    fake_label = torch.zeros((data_size, 1)).to(device)

    # Will be used for reporting
    real_losses, fake_losses, mse_losses, mae_losses = list(), list(), list(), list()
    real_losses_val, fake_losses_val, mse_losses_val, mae_losses_val = list(), list(), list(), list()

    real_losses2, fake_losses2, mse_losses2, mae_losses2 = list(), list(), list(), list()
    real_losses_val2, fake_losses_val2, mse_losses_val2, mae_losses_val2 = list(), list(), list(), list()

    k1_losses, k2_losses, k1_losses_val, k2_losses_val = list(), list(), list(), list()
    tp_losses_1_tr, tp_losses_1_val, tp_losses_2_tr, tp_losses_2_val = list(), list(), list(), list()
    gan_losses_1_tr, gan_losses_1_val, gan_losses_2_tr, gan_losses_2_val = list(), list(), list(), list()

    for epoch in range(num_epochs):
        # Reporting
        r, f, d, g, mse_l, mae_l = 0, 0, 0, 0, 0, 0
        r_val, f_val, d_val, g_val, mse_l_val, mae_l_val = 0, 0, 0, 0, 0, 0
        k1_train, k2_train, k1_val, k2_val = 0.0, 0.0, 0.0, 0.0
        r2, f2, d2, g2, mse_l2, mae_l2 = 0, 0, 0, 0, 0, 0
        r_val2, f_val2, d_val2, g_val2, mse_l_val2, mae_l_val2 = 0, 0, 0, 0, 0, 0
        tp1_tr, tp1_val, tp2_tr, tp2_val = 0.0, 0.0, 0.0, 0.0
        gan1_tr, gan1_val, gan2_tr, gan2_val = 0.0, 0.0, 0.0, 0.0

        # Train
        generator.train()
        discriminator.train()
        generator2.train()
        discriminator2.train()
        for i, data in enumerate(h_data_train_loader):
            data = data.to(device)

            optimizer_D.zero_grad()

            # Train the discriminator
            # Create fake data
            fake_y = generator(data).detach()
            edge_i, edge_a, _, _ = create_edge_index_attribute(fake_y)
            fake_data = Data(x=fake_y, edge_attr=edge_a, edge_index=edge_i).to(device)
            swapped_data = Data(x=data.y, edge_attr=data.y_edge_attr, edge_index=data.y_edge_index).to(device)

            # data: Real source and target
            # fake_data: Real source and generated target
            real_loss = adversarial_loss(discriminator(swapped_data, data), real_label[:data.x.size(0), :])
            fake_loss = adversarial_loss(discriminator(fake_data, data), fake_label[:data.x.size(0), :])
            loss_D = torch.mean(real_loss + fake_loss) / 2
            r += real_loss.item()
            f += fake_loss.item()
            d += loss_D.item()

            # Depending on the chosen training method, we might update the parameters of the discriminator
            if (epoch % 2 == 1 and opt.tr_st == "turns") or opt.tr_st == "same" or counter_d >= opt.id_e:
                loss_D.backward(retain_graph=True)
                optimizer_D.step()

            # Train the generator
            optimizer_G.zero_grad()

            # Adversarial Loss
            fake_data.x = generator(data)
            gan_loss = torch.mean(adversarial_loss(discriminator(fake_data, data), real_label[:data.x.size(0), :]))
            gan1_tr += gan_loss.item()

            # KL Loss
            kl_loss = kl.kl_divergence(normal.Normal(fake_data.x.mean(dim=1), fake_data.x.std(dim=1)),
                                       normal.Normal(data.y.mean(dim=1), data.y.std(dim=1))).sum()

            # Topology Loss
            tp_loss = tp(fake_data.x.sum(dim=-1), data.y.sum(dim=-1))
            tp1_tr += tp_loss.item()

            # Identity Loss is included in the end
            loss_G = i_coeff * identity_loss(generator(swapped_data),
                                             data.y) + g_coeff * gan_loss + kl_coeff * kl_loss + tp_coeff * tp_loss
            g += loss_G.item()
            if (epoch % 2 == 0 and opt.tr_st == "turns") or opt.tr_st == "same" or counter_g < opt.id_e:
                loss_G.backward(retain_graph=True)
                optimizer_G.step()
            k1_train += kl_loss.item()
            mse_l += msel(generator(data), data.y).item()
            mae_l += mael(generator(data), data.y).item()

            optimizer_D2.zero_grad()

            # Train the discriminator2

            # Create fake data for t2 from fake data for t1
            fake_data.x = fake_data.x.detach()
            fake_y2 = generator2(fake_data).detach()
            edge_i, edge_a, _, _ = create_edge_index_attribute(fake_y2)
            fake_data2 = Data(x=fake_y2, edge_attr=edge_a, edge_index=edge_i).to(device)
            swapped_data2 = Data(x=data.y2, edge_attr=data.y2_edge_attr, edge_index=data.y2_edge_index).to(device)

            # fake_data: Data generated for t1
            # fake_data2: Data generated for t2 using generated data for t1
            # swapped_data2: Real t2 data
            real_loss = adversarial_loss(discriminator2(swapped_data2, fake_data), real_label[:data.x.size(0), :])
            fake_loss = adversarial_loss(discriminator2(fake_data2, fake_data), fake_label[:data.x.size(0), :])
            loss_D = torch.mean(real_loss + fake_loss) / 2
            r2 += real_loss.item()
            f2 += fake_loss.item()
            d2 += loss_D.item()

            if (epoch % 2 == 1 and opt.tr_st == "turns") or opt.tr_st == "same" or counter_d >= opt.id_e:
                loss_D.backward(retain_graph=True)
                optimizer_D2.step()

            # Train generator2
            optimizer_G2.zero_grad()

            # Adversarial Loss
            fake_data2.x = generator2(fake_data)
            gan_loss = torch.mean(
                adversarial_loss(discriminator2(fake_data2, fake_data), real_label[:data.x.size(0), :]))
            gan2_tr += gan_loss.item()

            # Topology Loss
            tp_loss = tp(fake_data2.x.sum(dim=-1), data.y2.sum(dim=-1))
            tp2_tr += tp_loss.item()

            # KL Loss
            kl_loss = kl.kl_divergence(normal.Normal(fake_data2.x.mean(dim=1), fake_data2.x.std(dim=1)),
                                       normal.Normal(data.y2.mean(dim=1), data.y2.std(dim=1))).sum()

            # Identity Loss
            loss_G = i_coeff * identity_loss(generator(swapped_data2),
                                             data.y2) + g_coeff * gan_loss + kl_coeff * kl_loss + tp_coeff * tp_loss
            g2 += loss_G.item()
            if (epoch % 2 == 0 and opt.tr_st == "turns") or opt.tr_st == "same" or counter_g < opt.id_e:
                loss_G.backward(retain_graph=True)
                optimizer_G2.step()

            k2_train += kl_loss.item()
            mse_l2 += msel(generator2(fake_data), data.y2).item()
            mae_l2 += mael(generator2(fake_data), data.y2).item()

        # Validate
        generator.eval()
        discriminator.eval()
        generator2.eval()
        discriminator2.eval()

        for i, data in enumerate(h_data_test_loader):
            data = data.to(device)
            # Train the discriminator
            # Create fake data
            fake_y = generator(data).detach()
            edge_i, edge_a, _, _ = create_edge_index_attribute(fake_y)
            fake_data = Data(x=fake_y, edge_attr=edge_a, edge_index=edge_i).to(device)
            swapped_data = Data(x=data.y, edge_attr=data.y_edge_attr, edge_index=data.y_edge_index).to(device)

            # data: Real source and target
            # fake_data: Real source and generated target
            real_loss = adversarial_loss(discriminator(swapped_data, data), real_label[:data.x.size(0), :])
            fake_loss = adversarial_loss(discriminator(fake_data, data), fake_label[:data.x.size(0), :])
            loss_D = torch.mean(real_loss + fake_loss) / 2
            r_val += real_loss.item()
            f_val += fake_loss.item()
            d_val += loss_D.item()

            # Adversarial Loss
            fake_data.x = generator(data)
            gan_loss = torch.mean(adversarial_loss(discriminator(fake_data, data), real_label[:data.x.size(0), :]))
            gan1_val += gan_loss.item()

            # Topology Loss
            tp_loss = tp(fake_data.x.sum(dim=-1), data.y.sum(dim=-1))
            tp1_val += tp_loss.item()

            kl_loss = kl.kl_divergence(normal.Normal(fake_data.x.mean(dim=1), fake_data.x.std(dim=1)),
                                       normal.Normal(data.y.mean(dim=1), data.y.std(dim=1))).sum()

            # Identity Loss

            loss_G = i_coeff * identity_loss(generator(swapped_data), data.y) + g_coeff * gan_loss * kl_coeff * kl_loss
            g_val += loss_G.item()
            mse_l_val += msel(generator(data), data.y).item()
            mae_l_val += mael(generator(data), data.y).item()
            k1_val += kl_loss.item()

            # Second GAN

            # Create fake data for t2 from fake data for t1
            fake_data.x = fake_data.x.detach()
            fake_y2 = generator2(fake_data)
            edge_i, edge_a, _, _ = create_edge_index_attribute(fake_y2)
            fake_data2 = Data(x=fake_y2, edge_attr=edge_a, edge_index=edge_i).to(device)
            swapped_data2 = Data(x=data.y2, edge_attr=data.y2_edge_attr, edge_index=data.y2_edge_index).to(device)

            # fake_data: Data generated for t1
            # fake_data2: Data generated for t2 using generated data for t1
            # swapped_data2: Real t2 data
            real_loss = adversarial_loss(discriminator2(swapped_data2, fake_data), real_label[:data.x.size(0), :])
            fake_loss = adversarial_loss(discriminator2(fake_data2, fake_data), fake_label[:data.x.size(0), :])
            loss_D = torch.mean(real_loss + fake_loss) / 2
            r_val2 += real_loss.item()
            f_val2 += fake_loss.item()
            d_val2 += loss_D.item()

            # Adversarial Loss
            fake_data2.x = generator2(fake_data)
            gan_loss = torch.mean(
                adversarial_loss(discriminator2(fake_data2, fake_data), real_label[:data.x.size(0), :]))
            gan2_val += gan_loss.item()

            # Topology Loss
            tp_loss = tp(fake_data2.x.sum(dim=-1), data.y2.sum(dim=-1))
            tp2_val += tp_loss.item()

            # KL Loss
            kl_loss = kl.kl_divergence(normal.Normal(fake_data2.x.mean(dim=1), fake_data2.x.std(dim=1)),
                                       normal.Normal(data.y2.mean(dim=1), data.y2.std(dim=1))).sum()
            k2_val += kl_loss.item()

            # Identity Loss
            loss_G = i_coeff * identity_loss(generator(swapped_data2),
                                             data.y2) + g_coeff * gan_loss + kl_coeff * kl_loss
            g_val2 += loss_G.item()
            mse_l_val2 += msel(generator2(fake_data), data.y2).item()
            mae_l_val2 += mael(generator2(fake_data), data.y2).item()

        if opt.tr_st == 'idle':
            counter_g += 1
            counter_d += 1
            if counter_g == 2 * opt.id_e:
                counter_g = 0
                counter_d = 0

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(
            f'[Train]: D Loss: {d / total_step:.5f}, G Loss: {g / total_step:.5f} R Loss: {r / total_step:.5f}, F Loss: {f / total_step:.5f}, MSE: {mse_l / total_step:.5f}, MAE: {mae_l / total_step:.5f}')
        print(
            f'[Val]: D Loss: {d_val / val_step:.5f}, G Loss: {g_val / val_step:.5f} R Loss: {r_val / val_step:.5f}, F Loss: {f_val / val_step:.5f}, MSE: {mse_l_val / val_step:.5f}, MAE: {mae_l_val / val_step:.5f}')
        print(
            f'[Train]: D2 Loss: {d2 / total_step:.5f}, G2 Loss: {g2 / total_step:.5f} R2 Loss: {r2 / total_step:.5f}, F2 Loss: {f2 / total_step:.5f}, MSE: {mse_l2 / total_step:.5f}, MAE: {mae_l2 / total_step:.5f}')
        print(
            f'[Val]: D2 Loss: {d_val2 / val_step:.5f}, G2 Loss: {g_val2 / val_step:.5f} R2 Loss: {r_val2 / val_step:.5f}, F2 Loss: {f_val2 / val_step:.5f}, MSE: {mse_l_val2 / val_step:.5f}, MAE: {mae_l_val2 / val_step:.5f}')

        real_losses.append(r / total_step)
        fake_losses.append(f / total_step)
        mse_losses.append(mse_l / total_step)
        mae_losses.append(mae_l / total_step)
        real_losses_val.append(r_val / val_step)
        fake_losses_val.append(f_val / val_step)
        mse_losses_val.append(mse_l_val / val_step)
        mae_losses_val.append(mae_l_val / val_step)
        real_losses2.append(r2 / total_step)
        fake_losses2.append(f2 / total_step)
        mse_losses2.append(mse_l2 / total_step)
        mae_losses2.append(mae_l2 / total_step)
        real_losses_val2.append(r_val2 / val_step)
        fake_losses_val2.append(f_val2 / val_step)
        mse_losses_val2.append(mse_l_val2 / val_step)
        mae_losses_val2.append(mae_l_val2 / val_step)
        k1_losses.append(k1_train / total_step)
        k2_losses.append(k2_train / total_step)
        k1_losses_val.append(k1_val / val_step)
        k2_losses_val.append(k2_val / val_step)
        tp_losses_1_tr.append(tp1_tr / total_step)
        tp_losses_1_val.append(tp1_val / val_step)
        tp_losses_2_tr.append(tp2_tr / total_step)
        tp_losses_2_val.append(tp2_val / val_step)
        gan_losses_1_tr.append(gan1_tr / total_step)
        gan_losses_1_val.append(gan1_val / val_step)
        gan_losses_2_tr.append(gan2_tr / total_step)
        gan_losses_2_val.append(gan2_val / val_step)

    model_root = data_root

    if not os.path.exists(os.path.join(model_root, "/weights")):
            os.mkdir(os.path.join(model_root, "/weights"))

        # Save the models
    torch.save(generator.state_dict(), os.path.join(model_root, "/weights/generator_" + str(fold) + "_" + str(epoch) + "_" + str(opt.exp)))
    torch.save(discriminator.state_dict(),
                    os.path.join(model_root, "/discriminator_" + str(fold) + "_" + str(epoch) + "_" + str(opt.exp)))
    torch.save(generator2.state_dict(),
                    os.path.join(model_root, "/generator2_" + str(fold) + "_" + str(epoch) + "_" + str(opt.exp)))
    torch.save(discriminator2.state_dict(),
                    os.path.join(model_root, "/discriminator2_" + str(fold) + "_" + str(epoch) + "_" + str(opt.exp)))

    losses = [real_losses, fake_losses, mse_losses, mae_losses,
                  real_losses_val, fake_losses_val, mse_losses_val, mae_losses_val,
                  real_losses2, fake_losses2, mse_losses2, mae_losses2,
                  real_losses_val2, fake_losses_val2, mse_losses_val2, mae_losses_val2,
                  k1_losses, k2_losses, k1_losses_val, k2_losses_val,
                  tp_losses_1_tr, tp_losses_1_val, tp_losses_2_tr, tp_losses_2_val,
                  gan_losses_1_tr, gan_losses_1_val, gan_losses_2_tr, gan_losses_2_val
                  ]

    return losses


# Cross Validation
all_losses = ray.get([run_fold.remote(fold) for fold in range(folds)])

[real_losses, fake_losses, mse_losses, mae_losses,
 real_losses_val, fake_losses_val, mse_losses_val, mae_losses_val,
 real_losses2, fake_losses2, mse_losses2, mae_losses2,
 real_losses_val2, fake_losses_val2, mse_losses_val2, mae_losses_val2,
 k1_losses, k2_losses, k1_losses_val, k2_losses_val,
 tp_losses_1_tr, tp_losses_1_val, tp_losses_2_tr, tp_losses_2_val,
 gan_losses_1_tr, gan_losses_1_val, gan_losses_2_tr, gan_losses_2_val] = all_losses[-1]

# import os
# losses_dir = os.path.join(data_root, "losses")
# if not os.path.exists(losses_dir):
#   os.mkdir(losses_dir)

# with open(os.path.join(losses_dir, f"G_TrainLoss_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(gen_mae_losses_tr, f)
# with open(os.path.join(losses_dir, f"G_ValLoss_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(gen_mae_losses_val, f)
# with open(os.path.join(losses_dir, f"D_TrainRealLoss_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(disc_real_losses_tr, f)
# with open(os.path.join(losses_dir, f"D_TrainFakeLoss_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(disc_fake_losses_tr, f)
# with open(os.path.join(losses_dir, f"D_ValRealLoss_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(disc_real_losses_val, f)
# with open(os.path.join(losses_dir, f"D_ValFakeLoss_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(disc_fake_losses_val, f)
# with open(os.path.join(losses_dir, f"G2_TrainLoss_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(gen_mae_losses_tr2, f)
# with open(os.path.join(losses_dir, f"G2_ValLoss_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(gen_mae_losses_val2, f)
# with open(os.path.join(losses_dir, f"D2_TrainRealLoss_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(disc_real_losses_tr2, f)
# with open(os.path.join(losses_dir, f"D2_TrainFakeLoss_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(disc_fake_losses_tr2, f)
# with open(os.path.join(losses_dir, f"D2_ValRealLoss_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(disc_real_losses_val2, f)
# with open(os.path.join(losses_dir, f"D2_ValFakeLoss_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(disc_fake_losses_val2, f)
# with open(os.path.join(losses_dir, f"GenTotal_Train_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(gen_mae_losses_tr + gen_mae_losses_tr2, f)
# with open(os.path.join(losses_dir, f"GenTotal_Val_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(gen_mae_losses_val + gen_mae_losses_val2, f)
# with open(os.path.join(losses_dir, f"K1_TrainLoss_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(k1_train_s, f)
# with open(os.path.join(losses_dir, f"K1_ValLoss_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(k2_train_s, f)
# with open(os.path.join(losses_dir, f"K2_TrainLoss_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(k1_val_s, f)
# with open(os.path.join(losses_dir, f"K2_ValLoss_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(k2_val_s, f)
# with open(os.path.join(losses_dir, f"TP1_TrainLoss_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(tp1_train_s, f)
# with open(os.path.join(losses_dir, f"TP1_ValLoss_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(tp2_train_s, f)
# with open(os.path.join(losses_dir, f"TP2_TrainLoss_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(tp1_val_s, f)
# with open(os.path.join(losses_dir, f"TP2_ValLoss_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(tp2_val_s, f)
# with open(os.path.join(losses_dir, f"GAN1_TrainLoss_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(gan1_train_s, f)
# with open(os.path.join(losses_dir, f"GAN1_ValLoss_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(gan2_train_s, f)
# with open(os.path.join(losses_dir, f"GAN2_TrainLoss_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(gan1_val_s, f)
# with open(os.path.join(losses_dir, f"GAN2_ValLoss_exp_{opt.exp}"), "wb") as f:
#     pickle.dump(gan2_val_s, f)

print(f"Training Complete for experiment {opt.exp}!")


# COMMAND ----------


