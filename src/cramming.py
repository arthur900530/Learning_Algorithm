import copy
import torch
import torch.nn.functional as F


def get_unacceptable(model, dataloaders, v):
    train_loader = dataloaders['train']
    clone_model = copy.deepcopy(model).to('cpu')
    ua_cases = []
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.cpu()
            outputs = clone_model(inputs).squeeze(-1)
            outputs = outputs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            for i in range(outputs.shape[0]):
                if outputs[i] < v:
                    pred = 0
                else:
                    pred = 1
                if labels[i] != pred:
                    ua_cases.append((inputs[i],labels[i]))
    return ua_cases

def isolation(case, dataloaders, zeta = 1e-3):
    case = case.cpu().detach()                      # 12
    train_loader = dataloaders['train']
    while True:
        gamma_not_fit = False
        gamma = torch.rand(12,1)
        for inputs, labels in train_loader:            # 8,12  8,
            inputs = inputs.cpu().detach()
            for i in range(inputs.shape[0]):
                if not torch.equal(inputs[i], case):
                    if torch.matmul(torch.transpose(gamma, 0, 1),(inputs[i]-case)) != 0 and (zeta + torch.matmul(torch.transpose(gamma, 0, 1),(inputs[i]-case)))*(zeta - torch.matmul(torch.transpose(gamma, 0, 1),(inputs[i]-case))) < 0:
                        continue
                    else:
                        gamma_not_fit = True
                        break
            if gamma_not_fit:
                break
        if not gamma_not_fit:
            break
    length = torch.sqrt(torch.sum(torch.square(gamma)))
    gamma /= length
    return gamma


def cram(model, dataloaders, v, omin, zmax, zeta = 1e-3):
    cases = get_unacceptable(model, dataloaders, v)
    omin = torch.tensor(omin)
    zmax = torch.tensor(zmax)
    clone_model = copy.deepcopy(model).to('cpu')
    params = clone_model.state_dict()
    for case in cases:
        print('='*50)
        with torch.set_grad_enabled(False):
            former_product = torch.matmul(params['2.weight'], F.relu(torch.matmul(params['0.weight'],case[0])+params['0.bias']))
            # zero = torch.matmul(params['0.weight'][-3:], case[0]) + params['0.bias'][-3:]
            # print(zero)
            gamma = isolation(case[0], dataloaders, zeta)   # 12,1
            print(f'G: {gamma}, Case: {case[0]}')
            whp1 = whp2 = whp3 = torch.transpose(gamma, 0, 1)
            added_weights = torch.cat((whp1, whp2, whp3), 0)
            params['0.weight'] = torch.cat((params['0.weight'], added_weights), 0)

            whp1o = zeta - torch.matmul(torch.transpose(gamma, 0, 1), case[0])
            whp2o = - torch.matmul(torch.transpose(gamma, 0, 1), case[0])
            whp3o = -zeta - torch.matmul(torch.transpose(gamma, 0, 1), case[0])

            added_biases = torch.cat((whp1o, whp2o, whp3o), 0)

            params['0.bias'] = torch.cat((params['0.bias'], added_biases), 0)

            if case[1] == 1:
                wop1 = wop3 = (omin - params['2.bias'] - former_product)/zeta
                wop2 = -2 * (omin - params['2.bias'] - former_product)/zeta
                added_weights2 = torch.cat((wop1, wop2, wop3)).unsqueeze(0)
                params['2.weight'] = torch.cat((params['2.weight'], added_weights2), 1)
            else:
                wop1 = wop3 = (zmax - params['2.bias'] - former_product) / zeta
                wop2 = -2 * (zmax - params['2.bias'] - former_product) / zeta
                added_weights2 = torch.cat((wop1, wop2, wop3)).unsqueeze(0)
                params['2.weight'] = torch.cat((params['2.weight'], added_weights2), 1)

    return params


