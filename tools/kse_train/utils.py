#coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import random
import math
import numpy as np
import time
import pdb
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from math import isnan


def density_entropy(X): # 计算密度熵， X: N*C*D
    K = 5  # 经验值
    N, C, D = X.shape
    x = X.transpose(1, 0, 2).reshape(C, N, -1) # C*N*D
    score = []
    for c in range(C):
        nbrs = NearestNeighbors(n_neighbors=K + 1).fit(x[c]) # K近邻
        dms = []
        for i in range(N):
            dm = 0
            dist, ind = nbrs.kneighbors(x[c, i].reshape(1, -1))
            for j, id in enumerate(ind[0][1:]):
                dm += dist[0][j + 1]

            dms.append(dm)

        dms_sum = sum(dms)
        en = 0
        for i in range(N):
            en += -dms[i]/dms_sum*math.log(dms[i]/dms_sum, 2) # 计算熵

        score.append(en)
    return np.array(score)


class Conv2d_KSE(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, bias=True, G=4, T=0):
        super(Conv2d_KSE, self).__init__()

        self.output_channels = output_channels
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.isbias = bias
        self.T = T

        if G == 0:
            if output_channels >= input_channels:
                self.G = input_channels
            else:
                self.G = math.ceil(input_channels/output_channels)
        else:
            self.G = G
        self.group_num = self.G
        self.weight = nn.Parameter(torch.Tensor(output_channels, input_channels, kernel_size, kernel_size))
        self.mask = nn.Parameter(torch.Tensor(input_channels), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_channels))
        else:
            self.register_parameter('bias', None)

    def __repr__(self):
        return self.__class__.__name__ \
               + "({" + str(self.input_channels) \
               + "}, {" + str(self.output_channels) \
               + "}, kernel_size={" + str(self.kernel_size) + "}, stride={" + \
               str(self.stride) + "}, padding={" + str(self.padding) + "})"

    def forward(self, input):  # 16 16 3 3
        # transform cluster and index into weight for training
        cluster_weights = []
        for g in range(1, self.G - 1):
            if self.group_size[g] == 0:  # group_size = [2,2,7,5]  0 4 8 16 
                continue
            cluster = self.__getattr__("clusters_" + str(g))   # 4 * 2 * 3 *3
            clusters = cluster.permute(1, 0, 2, 3).contiguous().view(
                self.cluster_num[g] * cluster.shape[1], self.kernel_size, self.kernel_size)
            cluster_weight = clusters[
                self.__getattr__("cluster_indexs_" + str(g)).data].contiguous().view(
                self.output_channels, cluster.shape[1], self.kernel_size, self.kernel_size)  # 16 2 3 3，16 7 3 3

            cluster_weights.append(cluster_weight)

        if len(cluster_weights) == 0:
            weight = self.full_weight
        else:
            weight = torch.cat((self.full_weight, torch.cat(cluster_weights, dim=1)), dim=1) # 16, 14 3 3 self.full_weight 是全保留的

        return F.conv2d(torch.index_select(input, 1, self.channels_indexs), weight, self.bias, # input (bs)128，16，32,32 选择input的部分通道(不为0的)
                        stride=self.stride,
                        padding=self.padding)

    def KSE(self, G=None, T=None):
        if G is not None:
            self.G = G
        if T is not None:
            self.T = T
        weight = self.weight.data.cpu().numpy()

        # Calculate channels KSE indicator.
        weight = weight.transpose(1, 0, 2, 3).reshape(self.input_channels, self.output_channels, -1)  # C_out, C_in, K*K
        ks_weight = np.sum(np.linalg.norm(weight, ord=1, axis=2), 1)   # ks_weight = sum(|w|_1)
        ke_weight = density_entropy(weight.reshape(self.output_channels, self.input_channels, -1)) # ke_weight = entropy()
        ks_weight = (ks_weight - np.min(ks_weight)) / (np.max(ks_weight) - np.min(ks_weight))  # normalize
        ke_weight = (ke_weight - np.min(ke_weight)) / (np.max(ke_weight) - np.min(ke_weight))  # normalize
        indicator = np.sqrt(ks_weight / (1 + ke_weight))    # 
        indicator = (indicator - np.min(indicator))/(np.max(indicator) - np.min(indicator)) # normalize

        # Calculate each input channels kernel number and split them into different groups.
        # Each group has same kernel number.
        mask = np.zeros(shape=(self.input_channels))  # 记录每个in channel属于哪个group
        self.group_size = [0 for i in range(self.G)] # G=4 每个group的大小
        self.cluster_num = [1 for i in range(self.G)] # qc决定 每个group的 kernel数

        for i in range(self.input_channels):  # 计算groupsize
            if math.floor(indicator[i] * self.G) == 0: # 下取整，
                mask[i] = 0
                self.group_size[0] += 1
            elif math.ceil(indicator[i] * self.G) == self.G:
                mask[i] = self.G - 1
                self.group_size[-1] += 1
            else:
                mask[i] = math.floor(indicator[i] * self.G)  # 分类取决于 KSE indicator vc（[0,1]）的值。
                self.group_size[int(math.floor(indicator[i] * self.G))] += 1

        for i in range(self.G):
            if i == 0:
                self.cluster_num[i] = 0
            elif i == self.G - 1:
                self.cluster_num[i] = self.output_channels
            else:
                self.cluster_num[i] = math.ceil(self.output_channels * math.pow(2, i + 1 - self.T - self.G)) # 文中的qc 需要为该输入通道c 输出聚类数 保留几个kernel？
        self.mask.data = torch.Tensor(mask)

        # Generate corresponding cluster and index.
        # For kernel number = N: use full_weight rather than cluster&index
        self.full_weight = nn.Parameter(torch.Tensor(
            self.output_channels, self.group_size[-1], self.kernel_size, self.kernel_size),requires_grad=True)

        for g in range(1, self.G - 1):
            if self.group_size[g] == 0:
                continue
            else:
                cluster = nn.Parameter(torch.Tensor(
                    self.cluster_num[g], self.group_size[g], self.kernel_size, self.kernel_size), requires_grad=True)
                index = nn.Parameter(torch.zeros(self.output_channels, self.group_size[g]), requires_grad=False)
                self.__setattr__("clusters_" + str(g), cluster)
                self.__setattr__("indexs_" + str(g), index)

        # Calculate cluster and index by k-means
        # First, collect weight corresponding to each group(same kernel number)
        weight = self.weight.data.cpu().numpy()
        weight_group = []  # 分组后的weight （输入维度c上）
        for g in range(self.G):  # weight分组 到 weight_group
            if self.group_size[g] == 0:
                weight_group.append([])
                continue
            each_weight_group = []
            for c in range(self.input_channels):
                if mask[c] == g:  # 分类
                    each_weight_group.append(np.expand_dims(weight[:, c], 1))
            each_weight_group = np.concatenate(each_weight_group, 1)
            weight_group.append(each_weight_group)

        self.full_weight.data = torch.Tensor(weight_group[-1])

        for g in range(1, self.G - 1):
            if self.group_size[g] == 0:
                continue
            cluster_weight = weight_group[g]
            cluster_weight = cluster_weight.transpose((1, 0, 2, 3)).reshape((
                self.group_size[g], self.output_channels, -1))  # Cin_group * Cout * K*K

            clusters = []  # 
            indexs = []
            for c in range(cluster_weight.shape[0]):  # 在压缩后的每个Cin通道上进行聚类
                kmean = KMeans(n_clusters=self.cluster_num[g]).fit(cluster_weight[c]) # 对每个C_cin_group(每个group)进行聚类
                centroids = kmean.cluster_centers_ # 聚类中心
                assignments = kmean.labels_

                clusters.append(np.expand_dims(np.reshape(centroids, [
                    self.cluster_num[g], self.kernel_size, self.kernel_size]), 1)) # 最终得到 C_group * N_out * K*K
                indexs.append(np.expand_dims(assignments, 1)) # 存放

            clusters = np.concatenate(clusters, 1) # 每一类的weight，（8,5,3,3）
            indexs = np.concatenate(indexs, 1)  # 

            self.__getattr__("clusters_"+str(g)).data = torch.Tensor(clusters)
            self.__getattr__("indexs_"+str(g)).data = torch.Tensor(indexs)

    def forward_init(self):
        # record the channel index of each group

        full_index = []     # input channels index which kernel number = N
        cluster_indexs = [] # input channels index which kernel number != N && != 0 最终返回4个
        all_indexs = []     # input channels index which kernel number != 0

        for i, m in enumerate(self.mask.data):  # mask记录每个channel所在的G i：Cin m:G_index
            if m == self.G - 1:
                full_index.append(i)
                all_indexs.append(i)

        for g in range(1, self.G - 1):  # 每group,就是分类储存在不同的列表里，G=0 的三个直接不要了
            if self.group_size[g] == 0:
                cluster_indexs.append([])
                continue
            cluster_index = []
            for i, m in enumerate(self.mask.data):
                if m == g:  # Cin 通道，分类到 cluster_index
                    cluster_index.append(i)
                    all_indexs.append(i)
            cluster_indexs.append(cluster_index)

        self.channels_indexs = nn.Parameter(torch.zeros(self.input_channels - self.group_size[0]).long(),
                                            requires_grad=False)

        # transform index for training
        if self.full_weight.is_cuda:
            self.channels_indexs.data = torch.LongTensor(all_indexs).cuda()
            self.channel_indexs = []
            for g in range(1, self.G - 1):
                if self.group_size[g] == 0:
                    continue
                index = self.__getattr__("indexs_" + str(g))  # 2x16
                self.__setattr__("cluster_indexs_" + str(g), nn.Parameter(
                    (index.data + self.cluster_num[g] * torch.Tensor(
                        [i for i in range(self.group_size[g])]).view(1, -1).cuda()).view(-1).long(),
                    requires_grad=False))

        else:
            self.channels_indexs.data = torch.LongTensor(all_indexs)
            self.channel_indexs = []
            for g in range(1, self.G - 1):
                if self.group_size[g] == 0:
                    continue
                index = self.__getattr__("indexs_" + str(g))
                self.__setattr__("cluster_indexs_" + str(g), nn.Parameter(
                    (index.data + self.cluster_num[g] * torch.Tensor(
                        [i for i in range(self.group_size[g])]).view(1, -1)).view(-1).long(),
                    requires_grad=False))

    def create_arch(self, G=None, T=None):
        if G is not None:
            self.G = G
        if T is not None:
            self.T = T

        # create architecture (clusters and indexs) base on mask
        mask = self.mask.data.cpu().numpy()
        self.group_size = [0 for i in range(self.G)]
        self.cluster_num = [1 for i in range(self.G)]

        for i in range(self.input_channels):
            self.group_size[int(mask[i])] += 1

        for i in range(self.G):
            if i == 0:
                self.cluster_num[i] = 0
            elif i == self.G - 1:
                self.cluster_num[i] = self.output_channels
            else:
                self.cluster_num[i] = math.ceil(self.output_channels * math.pow(2, i + 1 - self.T - self.G))

        self.full_weight = nn.Parameter(
            torch.Tensor(self.output_channels, self.group_size[-1], self.kernel_size, self.kernel_size),
            requires_grad=True)

        for g in range(1, self.G - 1): # 0 和 G-1 不用新建 index clusters
            if self.group_size[g] == 0:
                continue
            else:
                cluster = nn.Parameter(torch.zeros(
                    self.cluster_num[g], self.group_size[g], self.kernel_size, self.kernel_size), requires_grad=True)
                index = nn.Parameter(torch.ByteTensor(math.ceil(
                    math.ceil(math.log(self.cluster_num[g], 2)) * self.output_channels * self.group_size[g] / 8)),
                    requires_grad=False)
                self.__setattr__("clusters_" + str(g), cluster) 
                self.__setattr__("indexs_" + str(g), index)

    def load(self):
        # tranform index
        for g in range(1, self.G - 1):
            if self.group_size[g] == 0:
                continue
            cluster = self.__getattr__("clusters_" + str(g))
            index = self.__getattr__("indexs_"+str(g))

            Q = cluster.shape[0]
            bits = math.ceil(math.log(Q, 2))
            indexs = index.data.cpu().numpy()
            new_b = ""
            f = '{0:08b}'

            for n, i in enumerate(indexs):
                if (self.output_channels*self.group_size[g]*bits)% 8 != 0 and n == indexs.shape[0]-1:
                    continue
                new_b += f.format(i)

            if (self.output_channels * self.group_size[g] * bits) % 8 != 0:
                va = (self.output_channels * self.group_size[g] * bits) % 8
                new_b += f.format(indexs[-1])[-va:]

            new_index = []

            for i in range(int(len(new_b)/bits)):
                b = new_b[i*bits:(i+1)*bits]
                v = float(int(b, 2))
                new_index.append(v)

            ni = torch.Tensor(new_index).view(self.output_channels, self.group_size[g])
            self.__delattr__("indexs_" + str(g))
            self.__setattr__("indexs_"+str(g), nn.Parameter(ni, requires_grad=False))

    def save(self):
        # tranform index
        for g in range(1, self.G - 1):
            if self.group_size[g] == 0:
                continue
            cluster = self.__getattr__("clusters_" + str(g))
            index = self.__getattr__("indexs_"+str(g))

            Q = cluster.shape[0]

            bits = math.ceil(math.log(Q, 2))

            new_b = ""
            indexs = index.data.cpu().numpy().reshape(-1)
            f = '{0:0'+str(bits)+'b}'
            for i in indexs:
                nb = f.format(int(i))
                new_b += nb

            new_index = []
            for i in range(int(len(new_b)*1.0/8)):
                new_value = int(new_b[i*8:(i+1)*8], 2)
                new_index.append(new_value)

            if len(new_b) % 8 != 0:
                new_value = int(new_b[int(len(new_b)*1.0/8)*8:], 2)
                new_index.append(new_value)

            self.__delattr__("indexs_"+str(g))
            self.__setattr__("indexs_"+str(g), nn.Parameter(torch.ByteTensor(new_index), requires_grad=False))
            self.__delattr__("cluster_indexs_" + str(g))
        self.__delattr__("channels_indexs")
        self.__delattr__("group_size")
        self.__delattr__("cluster_num")
        self.__delattr__("weight")
        if self.bias is not None:
            self.__delattr__("bias")
