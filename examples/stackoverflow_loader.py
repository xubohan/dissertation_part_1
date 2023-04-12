import os.path as osp
import shutil, os
import torch
from sklearn.metrics import roc_auc_score
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops, negative_sampling
import dgl
import pandas as pd
from sklearn import preprocessing
from torch_geometric.data import InMemoryDataset, download_url,extract_gz,extract_tar
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.utils import subgraph
import numpy as np
'''
@Author: Bohan Xu
@Date: 03/April/2023
'''
class stackoverflowLoader(InMemoryDataset):
    available_datasets = {
        'sx-stackoverflow':['sx-stackoverflow.txt.gz'],
        'sx-superuser':['sx-superuser.txt.gz'],
        'sx-askubuntu':['sx-askubuntu.txt.gz'],
        'sx-mathoverflow':['sx-mathoverflow.txt.gz'],
        'email-Eu-core-temporal':['email-Eu-core-temporal.txt.gz'],
        'CollegeMsg':['CollegeMsg.txt.gz']
    }

    def __init__(self,name,root='dataset',transform = None,pre_transform = None,meta_dict = None):
        self.name = name
        self.url = 'https://snap.stanford.edu/data'
        if meta_dict is None:
            self.dir_name = '_'.join(name.split('-')) 
            
            # check if previously-downloaded folder exists.
            # If so, use that one.
            if osp.exists(osp.join(root, self.dir_name + '_pyg')):
                self.dir_name = self.dir_name + '_pyg'

            self.original_root = root
            self.root = osp.join(root, self.dir_name)
            
            # master = pd.read_csv(os.path.join(os.path.dirname(__file__),'ROOT', 'master.csv'), index_col = 0)
            assert self.name in self.available_datasets.keys()

            # self.meta_info = master[self.name]
            
        else:
            self.dir_name = meta_dict['dir_path']
            self.original_root = ''
            self.root = meta_dict['dir_path']
            self.meta_info = meta_dict


        if osp.isdir(self.root) and (not osp.exists(osp.join(self.root, 'RELEASE_v' + str(self.meta_info['version']) + '.txt'))):
            print(self.name + ' has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.root)


        # self.download_name = self.meta_info['download_name'] ## name of downloaded file, e.g., tox21

        # self.num_tasks = int(self.meta_info['num tasks'])
        # self.task_type = self.meta_info['task type']
        # self.eval_metric = self.meta_info['eval metric']
        # self.__num_classes__ = int(self.meta_info['num classes'])
        # self.is_hetero = self.meta_info['is hetero'] == 'True'
        # self.binary = self.meta_info['binary'] == 'True'



        super(stackoverflowLoader,self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def num_classes(self):
        return self.__num_classes__
    
    def get_idx_split(self, split_type = None):
        if split_type is None:
            split_type = self.meta_info['split']
            
        path = osp.join(self.root, 'split', split_type)

        # short-cut if split_dict.pt exists
        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        train_idx = pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header = None).values.T[0]
        valid_idx = pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header = None).values.T[0]
        test_idx = pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header = None).values.T[0]

        return {'train': torch.tensor(train_idx, dtype = torch.long), 'valid': torch.tensor(valid_idx, dtype = torch.long), 'test': torch.tensor(test_idx, dtype = torch.long)}


    @property
    def raw_file_names(self):
        if self.binary:
            return ['data.npz']
        else:
            file_names = ['edge']
            if self.meta_info['has_node_attr'] == 'True':
                file_names.append('node-feat')
            if self.meta_info['has_edge_attr'] == 'True':
                file_names.append('edge-feat')
            return [file_name + '.csv.gz' for file_name in file_names]

    @property
    def processed_file_names(self):
        # return 'geometric_data_processed.pt'
        pass

    def _download(self):
        if osp.isdir(self.original_root) and len(os.listdir(self.original_root)) > 0:
            return

        os.makedirs(self.original_root)
        self.download()

    def download(self):
        for name in self.available_datasets[self.name]:
            path = download_url(f'{self.url}/{name}', self.original_root)
            if name.endswith('.tar.gz'):
                extract_tar(path, self.original_root)
            elif name.endswith('.gz'):
                extract_gz(path, self.original_root)
            os.unlink(path)

    def process(self):
        
        # Read data
        # data = pd.read_csv(osp.join(self.original_root, 'sx-stackoverflow.txt'), header=None, sep = ' ')
        # data = data.rename(columns = {0:'source', 1:'target', 2:'timestamp'})
        # data = data.sort_values(by = 'source')
        
        pass

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

if __name__ == '__main__':
    # pyg_dataset = stackoverflowLoader(name = 'sx-stackoverflow',root='filetest')
    pyg_dataset = stackoverflowLoader(name = 'sx-superuser')
    print(pyg_dataset[0])
   
                