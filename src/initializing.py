import regression_weight_tuning_EU_LG_UA as EU_LG_UA
import torch
import torch.nn as nn


def init_RELU_EU_LG_UA_Regression(m, p, criterion, dataloaders, dataset_sizes, device='cpu'):
    model = nn.Sequential(
        nn.Linear(m, p),
        nn.ReLU(),
        nn.Linear(m, 1)).to(device)
    train_result = EU_LG_UA.train_model(model, criterion, dataloaders, dataset_sizes, device,
                                        PATH='../weights/train_checkpoint.pt',
                                        epsilon=1e-6, num_epochs=50, n=1, show=True, v=0.6)

    return model, train_result