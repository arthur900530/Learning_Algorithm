import regression_weight_tuning_EU_LG_UA as EU_LG_UA
import torch.nn.functional as F
import torch.nn as nn


def init_RELU_EU_LG_UA_Regression(p, criterion, dataloaders, dataset_sizes, learning_goal, lgep, device='cpu'):
    m = next(iter(dataloaders['train']))[0].shape[-1]

    class slfn(nn.Module):
        def __init__(self, m, p):
            super(slfn, self).__init__()
            self.l1 = nn.Linear(m, p)
            self.l2 = nn.Linear(p, 1)
        def forward(self, x):
            out = F.relu(self.l1(x))
            out = self.l2(out)
            return out

    model = slfn(m,p).to(device)
    if learning_goal == 'lsc':
        train_result = EU_LG_UA.train_model_lsc(model, criterion, dataloaders, dataset_sizes, device,
                                                PATH='../weights/train_checkpoint.pt',
                                                lr_epsilon=1e-6, num_epochs=50, n=1, show=True, v=0.6)
    elif learning_goal == 'lgt1':
        train_result = EU_LG_UA.train_model_lgt1_reg(model, criterion, dataloaders, dataset_sizes, device,
                                                 PATH='../weights/train_checkpoint.pt',
                                                 lr_epsilon=1e-6, num_epochs=50, lgep=lgep, show=True)
    else:
        print('No such learning goal...')
        return None

    return model, train_result