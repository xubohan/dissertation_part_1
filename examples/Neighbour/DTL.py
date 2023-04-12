from glzip import CSR
from BPTree import BPTree
import pandas as pd
import os
import numpy as np
from collections import defaultdict
import random
import time
import memory_profiler
from memory_profiler import memory_usage
 
'''
@Author: Bohan Xu
@Date: 03/April/2023
'''

# v official PyG torch_geometric.loader.neighbor_sampler
# edge_index class, not sampler....
from typing import Callable, List, NamedTuple, Optional, Tuple, Union
import torch
from torch import Tensor
from torch_geometric.typing import SparseTensor

class Adj(NamedTuple):
    edge_index: torch.Tensor
    e_id: torch.Tensor
    size: torch.Tensor

    def to(self, *args, **kwargs):
        return Adj(self.edge_index.to(*args, **kwargs),
                   self.e_id.to(*args, **kwargs), self.size)

# ^ official PyG torch_geometric.loader.neighbor_sampler


class TimeDataLoaderAndSampler():
    def __init__(self, order):
        self.data = None
        self.csr = None
        self.bpt = BPTree(order=order)

        '''
        available_datasets = [
        'sx-superuser.txt',
        'sx-askubuntu.txt',
        'sx-mathoverflow.txt',
        'email-Eu-core-temporal.txt',
        'CollegeMsg.txt',
        'sx-stackoverflow.txt'
        '''

    def process(self, addr = 'sx-stackoverflow.txt', i = 100000):
        print('This step may consume a little bit time')
        self.data = pd.read_csv(os.path.join(os.path.abspath('..')+'/dataset', addr), header=None, sep = ' ')
        print('Done pd.read_csv')
        self.data = self.data.rename(columns = {0:'source', 1:'target', 2:'timestamp'})
        bptins = self.bpt.insert
        i: int = i
        self.csr = CSR(edge_index=torch.tensor([self.data['source'][:i],self.data['target'][:i]], dtype=torch.long))
        print('Done Graph Compression')
        a = map(lambda x,y,z : bptins((x,y), z),self.data['source'][:i],self.data['target'][:i],self.data['timestamp'][:i])
        list(a)
        print('Done Creating Index')
        # print('Done')


    # def time_range_sampler(self, source, time_range, nums):
    #     if (self.data is None) or (self.csr is None):
    #         print('Please process data first')
    #         return None

    #     possible_edge = list()
    #     for num_layer in nums:
    #         get_neighbour = self.csr.neighbors(source)

    #         pass 
        
    #     pass
    
    # def _rec_nodes(self, record,source, nums, iter_times):
    #     if left == []:
    #         return record
    #     else:
    #         ngb = self.neighbors(source.pop(0))
    #         edge_list = list(map(lambda x,y: (x,y), [source]*len(nbg), nbg))
            
    #         return self._rec_nodes(record+ edge_list,source+ngb)
    
    # using python rewrite code from jake, not using muti-thread
    def __reindex(self, inputs, outputs, output_counts):
        if inputs == [] or outputs == [] or output_counts == []:
            raise 'None value detected'
        out_map = defaultdict(int)
        frontier = []
        n_id = 0

        for input_val in inputs:
            out_map[input_val] = n_id
            n_id += 1
            frontier.append(input_val)

        for output in outputs:
            if output not in out_map:
                out_map[output] = n_id
                n_id += 1
                frontier.append(output)

        row_idx = []
        col_idx = []
        cnt = 0

        for i, input_val in enumerate(inputs):
            idx = out_map[input_val]
            for _ in range(output_counts[i]):
                row_idx.append(idx)
                col_idx.append(out_map[outputs[cnt]])
                cnt += 1

        return frontier, row_idx, col_idx

    def __reservior_sampling(self, inputs, k):
        res = []

        if len(inputs) <= k:
            res = inputs

        if len(inputs) > k: 
            res = inputs[:k]
            for i in range(k, len(inputs)):
                idx = random.randint(0, i)
                if idx < k:
                    res[idx] = inputs[i]
        return res

    # inspired by rust code from Jake
    def _time_sampler_kernel(self, inputs, k,timespan,sampling_strategy):
        # timespan = [min, max]
        res = ([],[])
        # print('inputs: ',inputs)
        for v in inputs:
            record = []
            # tim_rec = []
            ns = self.csr.neighbors(v)
            # user can change this sampling method
            # ns = list(np.random.choice(ns,k))
            for onode in ns:
                times =self.bpt.search((v,onode))
                if times == None:
                    continue
                for ti in times:
                    if ti >= timespan[0] and ti <= timespan[1]:
                        record.append(onode)
                        # tim_rec.append(ti) 
                        break
            # user can change this sampling method
            if record != []:
                if sampling_strategy == 0:
                    record = list(np.random.choice(record,k))
                if sampling_strategy == 1:
                    record = self.__reservior_sampling(record,k)
            d = len(record)
            res = (res[0]+ record.copy(), res[1] + [d]) #, res[2]+tim_rec)
        return res

    # @profile
    def time_sampler(self, size, timespan, input_nodes, sampling_strategy):
        nodes = list(input_nodes)
        batch_size = len(nodes)
        adjs = []

        for k in size:
            out,cnt = self._time_sampler_kernel(nodes, k, timespan,sampling_strategy)
            frontier, dst, src = self.__reindex(nodes,out,cnt)
            size_val = (len(frontier), len(nodes))
            adjs = [[[src, dst], [],size_val]] + adjs
            nodes = frontier
        nodes = torch.tensor(nodes)
        adjs = [adj for adj in map(lambda x: Adj(torch.tensor(x[0]), torch.tensor(x[1]), torch.tensor(x[2])), adjs)] 
        return nodes, batch_size, adjs




if __name__ == '__main__':
    record = dict()
    temp = TimeDataLoaderAndSampler(order=5)
    method = 1
    data_size = 63497050
    temp.process(i = data_size)
    
    # n_id, batch_size, adjs = temp.time_sampler([10,10],[1218497718, 1219222022],[72,39,380,101,234,431,455,423,271,219],1)
    start_time = time.time()
    n_id, batch_size, adjs = temp.time_sampler([10,10],[1218497718, 1219222022],[72,39,380,101,234,431,455,423,271,219],method)
    # 1218497718, 1219222022
    end_time = time.time()
    print("Execution time: ", format(end_time - start_time, '.5f')," seconds")
    mem_usage = memory_usage((temp.time_sampler,([10,10],[1218497718, 1219222022],[72,39,380,101,234,431,455,423,271,219],method)))
    print("Memory usage: ",format(max(mem_usage),'.5f')," MiB")
    ##
    ## This step may consume a little bit time
    ## Done pd.read_csv
    ## Done Graph Compression
    ## Done Creating Index
    ## Execution time: 0.00980520248413086 seconds
    ## Memory usage: 1973.453125 MiB
    ##

    # import networkx as nx
    # import matplotlib.pyplot as plt
    # from collections import defaultdict
    # from scipy.stats import pearsonr

    # G = nx.Graph()

    # edges = list(zip(temp.data['source'][:data_size], temp.data['target'][:data_size]))  # 原始图的边列表，格式为 [(source, target), ...]
    # G.add_edges_from(edges)

    # subgraph_edges = list(zip(n_id[adjs[0].edge_index[0]].tolist(), n_id[adjs[0].edge_index[1]].tolist())) + list(zip(n_id[adjs[1].edge_index[0]].tolist(), n_id[adjs[1].edge_index[1]].tolist()))

    # subG = nx.Graph()
    # subG.add_edges_from(subgraph_edges)

    # degree_sequence_G = sorted([d for n, d in G.degree()])
    # degree_sequence_subG = sorted([d for n, d in subG.degree()])
    # degree_freq_G = defaultdict(int)
    # degree_freq_subG = defaultdict(int)

    # for deg in degree_sequence_G:
    #     degree_freq_G[deg] += 1

    # for deg in degree_sequence_subG:
    #     degree_freq_subG[deg] += 1

    # degree_freq_G = {k: v / len(G) for k, v in degree_freq_G.items()}
    # degree_freq_subG = {k: v / len(subG) for k, v in degree_freq_subG.items()}

    # cc_G = nx.average_clustering(G)
    # cc_subG = nx.average_clustering(subG)

    # if nx.is_connected(G):
    #     diameter_G = nx.diameter(G)
    # else:
    #     print("Graph is not connected. Computing diameters for connected components.")
    #     diameter_G = [nx.diameter(G.subgraph(cc)) for cc in nx.connected_components(G)]
    #     print("Diameters of connected components: ", diameter_G)
    # diameter_subG = nx.diameter(subG)

    # print("Degree distribution:")
    # print("Original graph:", degree_freq_G)
    # print("Subgraph:", degree_freq_subG)

    # print("\nClustering coefficients:")
    # print("Original graph:", cc_G)
    # print("Subgraph:", cc_subG)

    # print("\nDiameters:")
    # print("Original graph:", diameter_G)
    # print("Subgraph:", diameter_subG)

    # common_degrees = set(degree_freq_G.keys()) & set(degree_freq_subG.keys())
    # x = [degree_freq_G[d] for d in common_degrees]
    # y = [degree_freq_subG[d] for d in common_degrees]
    # correlation, _ = pearsonr(x, y)

    # print("\nDegree distribution correlation:", correlation)
    # print("Difference in clustering coefficients:", abs(cc_G - cc_subG))
    # print("Difference in diameters:", abs(diameter_G - diameter_subG))



































