import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_unacceptable_lgt1_reg(model, dataloaders, lgep):
    train_loader = dataloaders['train']
    clone_model = copy.deepcopy(model).to('cpu')
    ua_cases = []
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.cpu()
            outputs = clone_model(inputs).squeeze(-1)
            # outputs = outputs.cpu().detach().numpy()
            # labels = labels.cpu().detach().numpy()
            for i in range(outputs.shape[0]):
                if torch.abs(outputs[i] - labels[i]) <= lgep:
                    pred = 1.0          # correctly predicted
                else:
                    pred = 0            # incorrectly predicted
                if pred == 0:
                    # print(f'Pred: {outputs[i].item()}, Label: {labels[i].item()}')
                    ua_cases.append((inputs[i],labels[i]))
    return ua_cases


def isolation(case, dataloaders, zeta):
    case = case.to('cpu')
    zeta = zeta.to('cpu')
    train_loader = dataloaders['train']
    gamma_found = False
    while not gamma_found:
        gamma_not_fit = False
        gamma = torch.rand(18, dtype=torch.float32)
        # gamma = F.normalize(gamma, p=2, dim=0, dtype=torch.float32)
        length = torch.sqrt(torch.sum(torch.square(gamma)))
        gamma = torch.div(gamma, length)
        for inputs, _ in train_loader:
            inputs = inputs.to('cpu')
            for i in range(inputs.shape[0]):
                if not torch.equal(inputs[i], case):
                    if torch.equal(torch.dot(gamma, (inputs[i]-case)), torch.zeros(1,dtype=torch.float32)):
                        gamma_not_fit = True
                    else:
                        if not torch.equal(torch.sign((zeta + torch.dot(gamma,(inputs[i]-case)))*(zeta - torch.dot(gamma,(inputs[i]-case)))), torch.sign(torch.tensor([-0.01]))):
                            gamma_not_fit = True
            if gamma_not_fit:
                break
        if not gamma_not_fit:
            gamma_found = True
    # length = torch.sqrt(torch.sum(torch.square(gamma)))
    return gamma


class slfn(nn.Module):
    def __init__(self, m, p):
        super(slfn, self).__init__()
        self.l1 = nn.Linear(m,p)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(p,1)
    def forward(self, x):
        out = self.relu(self.l1(x))
        out = self.l2(out)
        return out

def cram_lgt1(model, dataloaders, lgep, zeta = 1e-2):
    zeta = torch.tensor([zeta], dtype=torch.float32)
    cases = get_unacceptable_lgt1_reg(model, dataloaders, lgep)
    model.eval()
    clone_model = copy.deepcopy(model).to('cpu')
    params = clone_model.state_dict()
    m = next(iter(dataloaders['train']))[0].shape[-1]
    p = params['l1.weight'].shape[0]
    num = 0
    for case in cases:
        num += 1
        with torch.set_grad_enabled(False):
            modela = slfn(m,p).to('cpu')
            modela.load_state_dict(params)
            modela.eval()
            former_product = modela(case[0].unsqueeze(0)).squeeze(0)
            # print('=' * 50)
            # print('bc: ',former_product)
            # former_product = former_product.to(torch.float32)
            gamma = isolation(case[0], dataloaders, zeta)   # 12,1
            # print(f'G: {gamma}, Case: {case[0]}')
            whp1 = whp2 = whp3 = gamma.unsqueeze(0)
            added_weights = torch.cat((whp1, whp2, whp3), 0)

            params['l1.weight'] = torch.cat((params['l1.weight'], added_weights), 0)
            # print('l1.weight', params['l1.weight'])
            whp1o = zeta - torch.matmul(gamma, case[0]).unsqueeze(0)
            whp2o = - torch.matmul(gamma, case[0]).unsqueeze(0)
            whp3o = -zeta - torch.matmul(gamma, case[0]).unsqueeze(0)

            added_biases = torch.cat((whp1o, whp2o, whp3o), 0)

            params['l1.bias'] = torch.cat((params['l1.bias'], added_biases), 0)
            # print('l1.bias', params['l1.bias'])
            # former_product = params['l2.bias']
            wop1 = wop3 = ((case[1] - former_product) / zeta).unsqueeze(0)
            wop2 = (-2 * (case[1] - former_product) / zeta).unsqueeze(0)
            params['l2.weight'] = torch.cat((params['l2.weight'], wop1,wop2,wop3), 1)
            p += 3
            modelb = slfn(m, p).to('cpu')
            modelb.load_state_dict(params)
            modelb.eval()
            former_product2 = modelb(case[0].unsqueeze(0)).squeeze(0)
            # print('ac: ',former_product2)
            # print('t : ', case[1])
            # if np.round(former_product2,4) == case[1]:
            #     print(f'Ua case number {num} crammed succesfully: True')
            # else:
            #     print(f'Ua case number {num} crammed succesfully: False')
    return params


# def get_unacceptable_lsc(model, dataloaders, v):
#     train_loader = dataloaders['train']
#     clone_model = copy.deepcopy(model).to('cpu')
#     ua_cases = []
#     with torch.no_grad():
#         for inputs, labels in train_loader:
#             inputs = inputs.cpu()
#             outputs = clone_model(inputs).squeeze(-1)
#             outputs = outputs.cpu().detach().numpy()
#             labels = labels.cpu().detach().numpy()
#             for i in range(outputs.shape[0]):
#                 if outputs[i] < v:
#                     pred = 0
#                 else:
#                     pred = 1
#                 if labels[i] != pred:
#                     ua_cases.append((inputs[i],labels[i]))
#     return ua_cases


# def cram_lsc(model, dataloaders, v, omin, zmax, zeta = 1e-3):
#     cases = get_unacceptable_lsc(model, dataloaders, v)
#     omin = torch.tensor(omin)
#     zmax = torch.tensor(zmax)
#     clone_model = copy.deepcopy(model).to('cpu')
#     params = clone_model.state_dict()
#     for case in cases:
#         print('='*50)
#         with torch.set_grad_enabled(False):
#             former_product = clone_model(case[0]) - params['2.bias']
#             print(former_product)
#             # zero = torch.matmul(params['0.weight'][-3:], case[0]) + params['0.bias'][-3:]
#             # print(zero)
#             gamma = isolation(case[0], dataloaders, zeta)   # 12,1
#             # print(f'G: {gamma}, Case: {case[0]}')
#             whp1 = whp2 = whp3 = torch.transpose(gamma, 0, 1)
#             added_weights = torch.cat((whp1, whp2, whp3), 0)
#             params['0.weight'] = torch.cat((params['0.weight'], added_weights), 0)
#
#             whp1o = zeta - torch.matmul(torch.transpose(gamma, 0, 1), case[0])
#             whp2o = - torch.matmul(torch.transpose(gamma, 0, 1), case[0])
#             whp3o = -zeta - torch.matmul(torch.transpose(gamma, 0, 1), case[0])
#
#             added_biases = torch.cat((whp1o, whp2o, whp3o), 0)
#
#             params['0.bias'] = torch.cat((params['0.bias'], added_biases), 0)
#
#             if case[1] == 1:
#                 wop1 = wop3 = (omin - params['2.bias'] - former_product)/zeta
#                 wop2 = -2 * (omin - params['2.bias'] - former_product)/zeta
#                 added_weights2 = torch.cat((wop1, wop2, wop3)).unsqueeze(0)
#                 params['2.weight'] = torch.cat((params['2.weight'], added_weights2), 1)
#             else:
#                 wop1 = wop3 = (zmax - params['2.bias'] - former_product) / zeta
#                 wop2 = -2 * (zmax - params['2.bias'] - former_product) / zeta
#                 added_weights2 = torch.cat((wop1, wop2, wop3)).unsqueeze(0)
#                 params['2.weight'] = torch.cat((params['2.weight'], added_weights2), 1)
#
#     return params

