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
    case = case[0].cpu().detach()
    print(case.shape)
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

# def cram(model, dataloaders, v, device, zeta = 1e-3):
#     cases = get_unacceptable(model, dataloaders, v, device)
#     for case in cases:
#         gamma = isolation(case, dataloaders, zeta)

