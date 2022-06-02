import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class dataDataset(Dataset):
    def __init__(self, filepath, mode) -> None:
        # load csv data
        data = pd.read_csv(filepath, header=None)
        X = data.iloc[:, :-1].values

        y = data.iloc[:, -1].values
        # feature scaling
        # sc = StandardScaler()
        # X = sc.fit_transform(X)

        # convert to tensors
        self.X = torch.tensor(X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.long)
        self.mode = mode
        if self.mode == 'train':
            self.X = self.X[96:]
            self.y = self.y[96:]
        elif self.mode == 'val':
            self.X = self.X[:96]
            self.y = self.y[:96]
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

class dataDatasetMSE(Dataset):
    def __init__(self, data, mode, index) -> None:
        # load csv data
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # feature scaling
        sc = StandardScaler()
        X = sc.fit_transform(X)
        self.mode = mode
        if self.mode == 'train':
            self.X = X[index]
            self.y = y[index]
            # print(self.X.shape)
        elif self.mode == 'val':
            self.X = np.delete(X, index, 0)
            self.y = np.delete(y, index, 0)
            # print(self.X.shape)
        # convert to tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def get_data_mse(FOLDERNAME, mode):
    data = pd.read_csv(FOLDERNAME, header=None)
    index = np.random.choice(data.shape[0], int(data.shape[0] * 0.4), replace=False)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # feature scaling
    sc = StandardScaler()
    X = sc.fit_transform(X)

    if mode == 'train':
        X = X[index]
        y = y[index]
    elif mode == 'val':
        X = np.delete(X, index, 0)
        y = np.delete(y, index, 0)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return (X, y)

def read_data_mse(FOLDERNAME):
    datasets = {'train': get_data_mse(FOLDERNAME, 'train'),
                    'val': dataDatasetMSE(FOLDERNAME, 'val')}
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

    return datasets, dataset_sizes



def read_data(FOLDERNAME, batch_size = None, mode = 'mse'):
    if mode == 'cross_entropy':
      datasets = {'train': dataDataset(FOLDERNAME, 'train'), 'val': dataDataset(FOLDERNAME, 'val')}
      if batch_size == None:
        dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=len(datasets[x]), shuffle=True, num_workers=2) for x in ['train', 'val']}
        dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
      else:
        dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=2) for x in ['train', 'val']}
        dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

    if mode == 'mse':
      data = pd.read_csv(FOLDERNAME, header=None)
      index = np.random.choice(data.shape[0], int(data.shape[0] * 0.8), replace=False)
      mse_datasets = {'train': dataDatasetMSE(data, 'train', index), 'val': dataDatasetMSE(data, 'val', index)}
      if batch_size == None:
        dataloaders = {x: torch.utils.data.DataLoader(mse_datasets[x], batch_size=len(mse_datasets[x]), shuffle=True, num_workers=1) for x in ['train', 'val']}
        dataset_sizes = {x: len(mse_datasets[x]) for x in ['train', 'val']}
      else:
        dataloaders = {x: torch.utils.data.DataLoader(mse_datasets[x], batch_size=batch_size, shuffle=True, num_workers=1) for x in ['train', 'val']}
        dataset_sizes = {x: len(mse_datasets[x]) for x in ['train', 'val']}
      
    return dataloaders, dataset_sizes


def random_pick_data(FOLDERNAME, d_num = 20):
    for i in range(d_num):
        num = i + 1
        print(num)
        data = pd.read_csv(FOLDERNAME, header=None).to_numpy()
        # print(data.shape)
        index = np.random.choice(data.shape[0], int(data.shape[0] * 0.8), replace=False)
        data2 = data[index]
        # print(data2.shape)
        file_name = f'{num}_Random_picked_copper_forecasting_data.csv'
        np.savetxt(file_name, data2, delimiter=',', fmt='%f')

