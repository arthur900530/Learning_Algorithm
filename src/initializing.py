import regression_weight_tuning_EU_LG_UA as EU_LG_UA
import torch
import torch.nn as nn


def init_RELU_EU_LG_UA_Regression(m, p, criterion, dataloaders, dataset_sizes, learning_goal, device='cpu'):
    model = nn.Sequential(
        nn.Linear(m, p),
        nn.ReLU(),
        nn.Linear(m, 1)).to(device)
    train_result = None
    if learning_goal == 'lsc':
        train_result = EU_LG_UA.train_model_lsc(model, criterion, dataloaders, dataset_sizes, device,
                                                PATH='../weights/train_checkpoint.pt',
                                                epsilon=1e-6, num_epochs=50, n=1, show=True, v=0.6)
    elif learning_goal == 'lgt1':
        train_result = EU_LG_UA.train_model_lgt1(model, criterion, dataloaders, dataset_sizes, device,
                                                 PATH='../weights/train_checkpoint.pt',
                                                 epsilon=1e-6, num_epochs=50, lgep=0.3, show=True)
    else:
        print('No such learning goal...')
        return None

    return model, train_result