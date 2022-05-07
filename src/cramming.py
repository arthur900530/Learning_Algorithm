import copy
import torch
import numpy as np
import matplotlib.pyplot as plt

def get_unacceptable(model, dataloaders, v, device):
    train_loader = dataloaders['train']
    clone_model = copy.deepcopy(model)
    ua_cases = []
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            outputs = clone_model(inputs).squeeze(-1)
            outputs = outputs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            for i in range(outputs.shape[0]):
                if outputs[i] <= v:
                    pred = 0
                else:
                    pred = 1
                if labels[i] != pred:
                    ua_cases.append((inputs[i],labels[i]))
    return ua_cases

def isolation(case, dataloaders, zeta = 1e-3):
    case = case[0].cpu().detach()                     # 12
    train_loader = dataloaders['train']
    gamma_not_fit = False
    while True:
        gamma = torch.rand(12,1)
        for inputs, labels in train_loader:            # 12,8  8,
            inputs = inputs.cpu().detach()
            for i in range(inputs.shape[0]):
                if torch.sum(inputs[i] - case) != 0 and (zeta + torch.matmul(torch.transpose(gamma, 0, 1),(inputs[0]-case)))*(zeta - torch.matmul(torch.transpose(gamma, 0, 1), (inputs[0]-case))) < 0:
                    continue
                else:
                    gamma_not_fit = True
                    break
            if gamma_not_fit:
                break
            else:
                continue
        break
    return gamma

# 0.weight   torch.Size([50, 12]) <class 'torch.Tensor'>
# 0.bias   torch.Size([50]) <class 'torch.Tensor'>
# 2.weight   torch.Size([1, 50]) <class 'torch.Tensor'>
# 2.bias   torch.Size([1]) <class 'torch.Tensor'>

def cram(model, dataloaders, v, omin, zmax, device, zeta = 1e-3):
    cases = get_unacceptable(model, dataloaders, v, device)
    params = model.state_dict()
    for case in cases:
        with torch.no_grad(True):
            former_product = torch.matmul(params['2.weight'], (torch.matmul(params['0.weight'],case[0])+params['0.bias']))
            gamma = isolation(case[0], dataloaders, zeta)   # 12,1
            whp1 = whp2 = whp3 = torch.transpose(gamma, 0, 1)
            added_weights = torch.cat((whp1,whp2,whp3), 0)
            params['0.weight'] = torch.cat((params['0.weight'], added_weights), 0)
            whp1o = zeta - torch.matmul(torch.transpose(gamma, 0, 1), case[0])
            whp2o = - torch.matmul(torch.transpose(gamma, 0, 1), case[0])
            whp3o = -zeta - torch.matmul(torch.transpose(gamma, 0, 1), case[0])
            added_biases = torch.cat((whp1o, whp2o, whp3o), 0)
            params['0.bias'] = torch.cat((params['0.bias'], added_biases), 0)
            if case[1] == 1:
                wop1 = wop3 = (omin - params['2.bias'] - former_product)/zeta
                wop2 = -2*(omin - params['2.bias'] - former_product)/zeta
                added_weights2 = torch.cat((wop1, wop2, wop3), 1)
                params['2.weight'] = torch.cat((params['2.weight'], added_weights2), 1)
            else:
                wop1 = wop3 = (zmax - params['2.bias'] - former_product) / zeta
                wop2 = -2 * (zmax - params['2.bias'] - former_product) / zeta
                added_weights2 = torch.cat((wop1, wop2, wop3), 1)
                params['2.weight'] = torch.cat((params['2.weight'], added_weights2), 1)

    return params


