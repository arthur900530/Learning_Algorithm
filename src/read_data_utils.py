import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class dataDataset(Dataset):
    def __init__(self, filepath, mode) -> None:

        # load csv data
        data = pd.read_csv(filepath, header=None)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # feature scaling
        sc = StandardScaler()
        X = sc.fit_transform(X)

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
    def __init__(self, filepath, mode) -> None:

        # load csv data
        data = pd.read_csv(filepath, header=None)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # feature scaling
        sc = StandardScaler()
        X = sc.fit_transform(X)

        # convert to tensors
        self.X = torch.tensor(X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        self.mode = mode
        if self.mode == 'train':
            self.X = self.X[:400]
            self.y = self.y[:400]
        elif self.mode == 'val':
            self.X = self.X[400:]
            self.y = self.y[400:]
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

def read_data(FOLDERNAME, batch_size = None, mode = 'cross_entropy'):    
    if mode == 'cross_entropy':
      datasets = {'train': dataDataset(FOLDERNAME, 'train'), 'val': dataDataset(FOLDERNAME, 'val')}
      if batch_size == None:
        dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=len(datasets[x]), shuffle=True, num_workers=2) for x in ['train', 'val']}
        dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
      else:
        dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=2) for x in ['train', 'val']}
        dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}

    if mode == 'mse':
      mse_datasets = {'train': dataDatasetMSE(FOLDERNAME, 'train'), 'val': dataDatasetMSE(FOLDERNAME, 'val')}
      if batch_size == None:
        dataloaders = {x: torch.utils.data.DataLoader(mse_datasets[x], batch_size=len(mse_datasets[x]), shuffle=True, num_workers=2) for x in ['train', 'val']}
        dataset_sizes = {x: len(mse_datasets[x]) for x in ['train', 'val']}
      else:
        dataloaders = {x: torch.utils.data.DataLoader(mse_datasets[x], batch_size=batch_size, shuffle=True, num_workers=2) for x in ['train', 'val']}
        dataset_sizes = {x: len(mse_datasets[x]) for x in ['train', 'val']}
      
    return dataloaders, dataset_sizes

