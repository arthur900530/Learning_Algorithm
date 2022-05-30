import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import regression_weight_tuning_EU_LG_UA as EU_LG_UA
import read_data_utils as utils


class slfn(nn.Module):
    def __init__(self, m, p):
        super(slfn, self).__init__()
        self.l1 = nn.Linear(m,p)
        self.l2 = nn.Linear(p,1)
    def forward(self, x):
        out = F.relu(self.l1(x))
        out = self.l2(out)
        return out


def learning_goal_1(model, dataloader, ep, device):
    clone_model = copy.deepcopy(model)
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = clone_model(inputs).squeeze(-1)
            outputs = outputs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            for i in range(outputs.shape[0]):
                if np.abs(i - labels[i]) <= ep:
                    continue
                else:
                    return False
    return True

def predict(outputs, lgep, labels, device):
    pred = torch.zeros(outputs.shape[0]).to(device)
    for i in range(outputs.shape[0]):
        if np.abs(outputs[i].item() - labels[i].item()) <= lgep:
            pred[i] = 1.0
        else:
            pred[i] = 0
    return pred

def reg_model_lgt1_3(model, criterion, dataloaders, dataset_sizes, device, PATH='../weights/reg_checkpoint.pt',
             num_epochs=10, lr=1e-5, lr_epsilon=1e-10, lgep=0.5, rs=0.001, show=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=rs)
    train_acc_history = []
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []
    lr_epsilon = lr_epsilon
    goal_achieved = True
    tiny_lr = False
    checkpoint = None
    epoch = 0
    lgep = lgep
    while epoch < num_epochs:
        torch.save({'model_state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict()}, PATH)
        checkpoint = torch.load(PATH)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            corrects = 0
            for inputs, labels in dataloaders[phase]:
                s = inputs.size(0)
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # forward
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(-1), labels)
                    pred = predict(outputs, lgep, labels, device)
                    # statistics
                    running_loss += loss.item() * s
                    # corrects += torch.sum(pred == labels.flatten().data)
                    corrects += torch.sum(pred).item()
                    if phase == 'train':
                        # backward & adjust weights
                        loss.backward()
                        optimizer.step()
            former_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(corrects) / dataset_sizes[phase]

            for inputs, labels in dataloaders[phase]:
                s = inputs.size(0)
                inputs = inputs.to(device)
                labels = labels.to(device)
                with torch.set_grad_enabled(False):
                    # forward
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(-1), labels)
                    # statistics
                    running_loss += loss.item() * s
            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                # learning goal
                if epoch_loss <= former_loss:
                    goal_achieved = learning_goal_1(model, dataloaders['train'], lgep, device)
                    if not goal_achieved:
                        print('not achieved')
                        break
                    else:
                        print('goal')
                        optimizer.param_groups[0]['lr'] *= 1.2
                        torch.save({'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict()}, PATH)
                        former_loss = epoch_loss
                        checkpoint = torch.load(PATH)
                        epoch += 1
                else:
                    if optimizer.param_groups[0]['lr'] < lr_epsilon:
                        tiny_lr = True
                        break
                    else:
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model.train()
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        optimizer.param_groups[0]['lr'] *= 0.1
                        torch.save({'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict()}, PATH)
                        checkpoint = torch.load(PATH)
                        break
                print('Epoch {}/{}'.format(epoch, num_epochs))
                print('-' * 10)
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
            else:
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            # if show:
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        if (not goal_achieved) or tiny_lr:
            break

    result_dict = {'model': checkpoint['model_state_dict'],
                   'train_loss_history': train_loss_history,
                   'val_loss_history': val_loss_history,
                   'train_acc_history': train_acc_history,
                   'val_acc_history': val_acc_history,
                   'lgep': lgep,
                   'lr': optimizer.param_groups[0]['lr']}

    if not goal_achieved:
        result_dict['result'] = False
        result_dict['msg'] = 'goal_failed'
    elif tiny_lr:
        result_dict['result'] = False
        result_dict['msg'] = 'tiny learning rate'
    else:
        result_dict['result'] = False
        result_dict['msg'] = 'trained all epoch'
    if show:
        plt.figure(figsize=(16, 14))
        plt.subplot(2, 1, 1)
        plt.plot(result_dict['train_loss_history'], '-o')
        plt.plot(result_dict['val_loss_history'], '-o')
        plt.legend(['train', 'val'], loc='lower right')
        plt.xlabel('iteration', fontsize=20)
        plt.ylabel('loss', fontsize=20)

        plt.subplot(2, 1, 2)
        plt.yticks([0.1, 0.3, 0.5, 0.7, 0.9])
        plt.plot(result_dict['train_acc_history'], '-o')
        plt.plot(result_dict['val_acc_history'], '-o')
        plt.legend(['train', 'val'], loc='lower right')
        plt.ylabel('accuracy', fontsize=20)
        plt.figure(figsize=(4, 4))
        plt.show()

    return result_dict


def reg_model_lgt1_2(model, criterion, dataloaders, dataset_sizes, device, PATH='../weights/reg_checkpoint.pt',
             num_epochs=10, lr=1e-5, lr_epsilon=1e-10, lgep=0.5, rs=0.001, show=False):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=rs)
    train_acc_history = []
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []
    lr_epsilon = lr_epsilon
    goal_achieved = True
    tiny_lr = False
    checkpoint = None
    epoch = 0
    lgep = lgep

    while epoch < num_epochs:
        torch.save({'model_state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(),
                   'former_loss': float("inf")}, PATH)
        checkpoint = torch.load(PATH)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            corrects = 0
            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                s = inputs.size(0)
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # forward
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(-1), labels)
                    pred = predict(outputs, lgep, labels, device)
                    # statistics
                    running_loss += loss.item() * s
                    # corrects += torch.sum(pred == labels.flatten().data)
                    corrects += torch.sum(pred).item()
                    if phase == 'train':
                        # backward & adjust weights
                        loss.backward()
                        optimizer.step()

            # statistics
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(corrects) / dataset_sizes[phase]

            if phase == 'train':
                # learning goal
                goal_achieved = learning_goal_1(model, dataloaders['train'], lgep, device)
                if not goal_achieved:
                    print('not achieved')
                    break
                else:
                    print('goal')
                # weight tuning
                if epoch_loss < checkpoint['former_loss']:
                    optimizer.param_groups[0]['lr'] *= 1.2
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'former_loss': epoch_loss}, PATH)
                    checkpoint = torch.load(PATH)

                else:
                    if optimizer.param_groups[0]['lr'] < lr_epsilon:
                        tiny_lr = True
                        break
                    else:
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model.train()
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        epoch_loss = checkpoint['former_loss']
                        optimizer.param_groups[0]['lr'] *= 0.7
                        torch.save({'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'former_loss': epoch_loss}, PATH)
                        checkpoint = torch.load(PATH)
                        # print('lr decrease')
                        break
                epoch += 1
                print('Epoch {}/{}'.format(epoch, num_epochs))
                print('-' * 10)
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
            else:
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            # if show:
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        if (not goal_achieved) or tiny_lr:
            break

    result_dict = {'model': checkpoint['model_state_dict'],
                   'train_loss_history': train_loss_history,
                   'val_loss_history': val_loss_history,
                   'train_acc_history': train_acc_history,
                   'val_acc_history': val_acc_history,
                   'lgep': lgep,
                   'lr': optimizer.param_groups[0]['lr']}

    if not goal_achieved:
        result_dict['result'] = False
        result_dict['msg'] = 'goal_failed'
    elif tiny_lr:
        result_dict['result'] = False
        result_dict['msg'] = 'tiny learning rate'
    else:
        result_dict['result'] = False
        result_dict['msg'] = 'trained all epoch'
    if show:
        plt.figure(figsize=(16, 14))
        plt.subplot(2, 1, 1)
        plt.plot(result_dict['train_loss_history'], '-o')
        plt.plot(result_dict['val_loss_history'], '-o')
        plt.legend(['train', 'val'], loc='lower right')
        plt.xlabel('iteration', fontsize=20)
        plt.ylabel('loss', fontsize=20)

        plt.subplot(2, 1, 2)
        plt.yticks([0.1, 0.3, 0.5, 0.7, 0.9])
        plt.plot(result_dict['train_acc_history'], '-o')
        plt.plot(result_dict['val_acc_history'], '-o')
        plt.legend(['train', 'val'], loc='lower right')
        plt.ylabel('accuracy', fontsize=20)
        plt.figure(figsize=(4, 4))
        plt.show()

    return result_dict


def predict(outputs, lgep, labels, device):
    pred = torch.zeros(outputs.shape[0]).to(device)
    for i in range(outputs.shape[0]):
        if np.abs(outputs[i].item() - labels[i].item()) <= lgep:
            pred[i] = 1.0
        else:
            pred[i] = 0
    # print(pred)
    return pred


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
                    pred = 1.0  # correctly predicted
                else:
                    pred = 0  # incorrectly predicted
                if pred == 0:
                    # print(f'Pred: {outputs[i].item()}, Label: {labels[i].item()}')
                    ua_cases.append((inputs[i], labels[i]))
    return ua_cases

# reorganize function below include regularization, weight-tunning and prunning hidden node
def reorg_model(model, criterion, device,PATH='../weights/reorg_checkpoint.pt',
                lr=1e-5, lr_epsilon=1e-10, lgep=0.5,rs=0.001):
    print('='*50)
    print("Start reorginization")
    p = model.state_dict()['l1.weight'].shape[0]
    print(f'Init p: {p}')
    k = 0
    # get dataloader with whole batch
    file = 'Copper_forecasting_data3.csv'
    dataloaders, dataset_sizes = utils.read_data(file, batch_size=None, mode='mse')

    while k < p:
        print("Start regularization")
        result = reg_model_lgt1_3(model, criterion, dataloaders, dataset_sizes, device, PATH='../weights/reg_checkpoint.pt',
                                num_epochs=20, lr=1e-8, lr_epsilon=lr_epsilon, lgep=lgep, rs=rs)
        print(result['msg'])
        print("Finish regularization")
        # store model state dict
        torch.save(result['model'], PATH)
        model.load_state_dict(torch.load(PATH))
        msd = model.state_dict()

        for key in msd.keys():
            if key == 'l1.weight':
                msd[key] = msd[key][[x for i, x in enumerate(range(p)) if i != k], :]
            elif key == 'l1.bias':
                msd[key] = msd[key][[x for i, x in enumerate(range(p)) if i != k]]
            elif key == 'l2.weight':
                msd[key] = msd[key][:, [x for i, x in enumerate(range(p)) if i != k]]

        m = next(iter(dataloaders['train']))[0].shape[-1]
        model = slfn(m, p-1).to(device)
        model.load_state_dict(msd)

        # print accuracy before weight-tuning
        # for phase in ['train', 'val']:
        #     corrects = 0
        #     for inputs, labels in dataloaders[phase]:
        #         inputs = inputs.to(device)
        #         labels = labels.to(device)
        #         with torch.set_grad_enabled(False):
        #             outputs = model(inputs)
        #             pred = predict(outputs, lgep, labels, device)
        #             corrects += torch.sum(pred).item()
        #     epoch_acc = float(corrects) / dataset_sizes[phase]
        #     print(f'Bwt {phase} accuracy： {epoch_acc}')

        print('Start weight-tuning')
        result = EU_LG_UA.train_model_lgt1_reg(model, criterion, dataloaders, dataset_sizes, device,
                                                 PATH='../weights/train_checkpoint.pt',
                                                 lr=lr, lr_epsilon=lr_epsilon, num_epochs=50, lgep=lgep, show=False)
        print('Finish weight-tuning')

        ua_cases = get_unacceptable_lgt1_reg(model, dataloaders, lgep)
        if len(ua_cases) <= 3:
            print(ua_cases)
        print(result['msg'])
        if result['result']:
            print('p--')
            p -= 1
            torch.save(model.state_dict(), PATH)
        else:
            print('k++')
            k += 1
            model = slfn(m, p).to(device)
            model.load_state_dict(torch.load(PATH))
        print(f'k: {k}, p: {p}')
        print('='*50)

    for phase in ['train', 'val']:
        corrects = 0
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                pred = predict(outputs, lgep, labels, device)
                corrects += torch.sum(pred).item()
        epoch_acc = float(corrects) / dataset_sizes[phase]
        print(f'{phase} accuracy： {epoch_acc}')

    return model.load_state_dict(torch.load(PATH)), p

# def reg_model_lgt1(model, criterion, dataloaders, dataset_sizes, device, PATH='../weights/reg_checkpoint.pt',
#              num_epochs=10, lr=1e-5, lr_epsilon=1e-10, lgep=0.5, rs=0.001, show=False):
#
#     def cal_reg_term(model, m, rs):
#         p = model.state_dict()['l1.weight'].shape[0]
#         params = 0
#         for p in model.parameters():
#             params += torch.sum(torch.square(p))
#         l = torch.tensor([rs/(p + 1 + p * (m + 1))], dtype=torch.float32).to(device)
#         reg_term = torch.mul(l, params)
#         return reg_term
#
#     def predict(outputs, lgep, labels, device):
#         pred = torch.zeros(outputs.shape[0]).to(device)
#         for i in range(outputs.shape[0]):
#             if np.abs(outputs[i].item() - labels[i].item()) <= lgep:
#                 pred[i] = 1.0
#             else:
#                 pred[i] = 0
#         # print(pred)
#         return pred
#
#     # dataloaders, dataset_sizes = utils.read_data(file, batch_size=None, mode='mse')
#     m = next(iter(dataloaders['train']))[0].shape[-1]
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     train_acc_history = []
#     train_loss_history = []
#     val_loss_history = []
#     val_acc_history = []
#     lr_epsilon = lr_epsilon
#     goal_achieved = True
#     tiny_lr = False
#     checkpoint = None
#     epoch = 0
#     lgep = lgep
#
#     while epoch < num_epochs:
#         torch.save({'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'former_loss': float("inf")}, PATH)
#         checkpoint = torch.load(PATH)
#         # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()
#             else:
#                 model.eval()
#             running_loss = 0.0
#             corrects = 0
#             # Iterate over data(one batch only)
#             for inputs, labels in dataloaders[phase]:
#                 s = inputs.size(0)
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#                 optimizer.zero_grad()
#
#                 with torch.set_grad_enabled(phase == 'train'):
#                     # forward
#                     outputs = model(inputs)
#                     reg_term = cal_reg_term(model, m, rs)
#                     # print('reg: ', reg_term)
#                     loss = criterion(outputs.squeeze(-1), labels) + reg_term
#                     # print('loss: ', loss)
#                     pred = predict(outputs, lgep, labels, device)
#                     # statistics
#                     running_loss += loss.item() * s
#                     # corrects += torch.sum(pred == labels.flatten().data)
#                     corrects += torch.sum(pred).item()
#                     if phase == 'train':
#                         # backward & adjust weights
#                         loss.backward()
#                         optimizer.step()
#
#             # statistics
#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = float(corrects) / dataset_sizes[phase]
#
#             if phase == 'train':
#                 # learning goal
#                 goal_achieved = learning_goal_1(model, dataloaders['train'], lgep, device)
#                 if not goal_achieved:
#                     break
#                 # weight tuning
#                 if epoch_loss < checkpoint['former_loss']:
#                     optimizer.param_groups[0]['lr'] *= 1.2
#                     torch.save({'model_state_dict': model.state_dict(),
#                                 'optimizer_state_dict': optimizer.state_dict(),
#                                 'former_loss': epoch_loss}, PATH)
#                     checkpoint = torch.load(PATH)
#                 else:
#                     if optimizer.param_groups[0]['lr'] < lr_epsilon:
#                         tiny_lr = True
#                         break
#                     else:
#                         model.load_state_dict(checkpoint['model_state_dict'])
#                         model.train()
#                         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#                         epoch_loss = checkpoint['former_loss']
#                         optimizer.param_groups[0]['lr'] *= 0.7
#                         torch.save({'model_state_dict': model.state_dict(),
#                                     'optimizer_state_dict': optimizer.state_dict(),
#                                     'former_loss': epoch_loss}, PATH)
#                         checkpoint = torch.load(PATH)
#                         # print('lr decrease')
#                         break
#                 epoch += 1
#                 print('Epoch {}/{}'.format(epoch, num_epochs))
#                 print('-' * 10)
#                 train_acc_history.append(epoch_acc)
#                 train_loss_history.append(epoch_loss)
#             else:
#                 val_acc_history.append(epoch_acc)
#                 val_loss_history.append(epoch_loss)
#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
#
#         if (not goal_achieved) or tiny_lr:
#             break
#
#     result_dict = {'model': checkpoint['model_state_dict'],
#                'train_loss_history': train_loss_history,
#                'val_loss_history': val_loss_history,
#                'train_acc_history': train_acc_history,
#                'val_acc_history': val_acc_history,
#                'lgep': lgep}
#
#     if not goal_achieved:
#         result_dict['result'] = False
#         result_dict['msg'] = 'goal_failed'
#     elif tiny_lr:
#         result_dict['result'] = False
#         result_dict['msg'] = 'tiny learning rate'
#     else:
#         result_dict['result'] = False
#         result_dict['msg'] = 'trained all epoch'
#     if show:
#         plt.figure(figsize=(16, 14))
#         plt.subplot(2, 1, 1)
#         plt.plot(result_dict['train_loss_history'], '-o')
#         plt.plot(result_dict['val_loss_history'], '-o')
#         plt.legend(['train', 'val'], loc='lower right')
#         plt.xlabel('iteration', fontsize=20)
#         plt.ylabel('loss', fontsize=20)
#
#         plt.subplot(2, 1, 2)
#         plt.yticks([0.1, 0.3, 0.5, 0.7, 0.9])
#         plt.plot(result_dict['train_acc_history'], '-o')
#         plt.plot(result_dict['val_acc_history'], '-o')
#         plt.legend(['train', 'val'], loc='lower right')
#         plt.ylabel('accuracy', fontsize=20)
#         plt.figure(figsize=(4, 4))
#         plt.show()
#     return result_dict

# def reg_model(model, criterion, dataloaders, dataset_sizes, device, PATH='../weights/reg_checkpoint.pt',
#               num_epochs=10, lr_epsilon=1e-8, lgep=0.3, rs=0.001, show=False):
#
#     def cal_reg_term(model, rs=0.001):
#         m = next(iter(dataloaders['train']))[0].shape[-1]
#         params = model.state_dict()
#         p = params['l1.weight'].shape[0]
#         params = 0
#         for p in model.parameters():
#             params += torch.sum(p ** 2)
#         reg_term = (rs / (p + 1 + p * (m + 1))) * params
#         print('reg', reg_term)
#         return reg_term
#
#     def predict(outputs, ep, labels, device):
#         pred = torch.zeros(outputs.shape[0]).to(device)
#         for i in range(outputs.shape[0]):
#             if np.abs(outputs[i].item() - labels[i].item()) <= ep:
#                 pred[i] = 1.0
#             else:
#                 pred[i] = 0
#         return pred
#
#     # local variables
#     optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
#     train_acc_history = []
#     train_loss_history = []
#     val_loss_history = []
#     val_acc_history = []
#     lr_epsilon = lr_epsilon
#     goal_failed = False
#     tiny_lr = False
#     checkpoint = None
#     epoch = -1
#
#     while epoch < num_epochs:
#         torch.save({'model_state_dict': model.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict()}, PATH)
#         checkpoint = torch.load(PATH)
#         # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             running_loss = 0.0
#             corrects = 0
#             if phase == 'train':
#                 model.train()
#             else:
#                 model.eval()
#             if epoch == -1:
#                 for inputs, labels in dataloaders[phase]:
#                     s = inputs.size(0)
#                     inputs = inputs.to(device)
#                     labels = labels.to(device)
#                     with torch.set_grad_enabled(phase == 'train'):
#                         optimizer.zero_grad()
#                         # forward
#                         outputs = model(inputs)
#                         reg_term = cal_reg_term(model, rs)
#                         loss = criterion(outputs.squeeze(-1), labels) + reg_term
#                         print(loss)
#                         # statistics
#                         running_loss += loss.item() * s
#                         if phase == 'train':
#                             # backward & adjust weights
#                             loss.backward()
#                             optimizer.step()
#                 epoch_loss = running_loss / dataset_sizes[phase]
#                 goal_achieved = learning_goal_1(model, dataloaders['train'], lgep, device)
#                 if goal_achieved:
#                     former_loss = epoch_loss
#                     epoch += 1
#                     print('a')
#                     break
#                 else:
#                     goal_failed = True
#                     break
#
#             # Iterate over data
#             for inputs, labels in dataloaders[phase]:
#                 s = inputs.size(0)
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#
#                 with torch.set_grad_enabled(phase == 'train'):
#                     # forward
#                     optimizer.zero_grad()
#                     outputs = model(inputs)
#                     reg_term = cal_reg_term(model, rs)
#                     loss = criterion(outputs.squeeze(-1), labels) + reg_term
#                     pred = predict(outputs, lgep, labels, device)
#                     # statistics
#                     running_loss += loss.item() * s
#                     corrects += torch.sum(pred).item()
#                     if phase == 'train':
#                         # backward & adjust weights
#                         loss.backward()
#                         optimizer.step()
#
#             # statistics
#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = float(corrects) / dataset_sizes[phase]
#
#             if phase == 'train':
#                 # weight tuning
#                 if epoch_loss <= former_loss:
#                     # learning goal
#                     goal_achieved = learning_goal_1(model, dataloaders['train'], lgep, device)
#                     if goal_achieved:
#                         optimizer.param_groups[0]['lr'] *= 1.2
#                         torch.save({'model_state_dict': model.state_dict(),
#                                     'optimizer_state_dict': optimizer.state_dict()}, PATH)
#                         checkpoint = torch.load(PATH)
#                         former_loss = epoch_loss
#                     else:
#                         goal_failed = True
#                 else:
#                     if optimizer.param_groups[0]['lr'] < lr_epsilon:
#                         tiny_lr = True
#                     else:
#                         model.load_state_dict(checkpoint['model_state_dict'])
#                         model.train()
#                         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#                         optimizer.param_groups[0]['lr'] *= 0.7
#                         torch.save({'model_state_dict': model.state_dict(),
#                                     'optimizer_state_dict': optimizer.state_dict()}, PATH)
#                         checkpoint = torch.load(PATH)
#                         break
#                 epoch += 1
#                 print('Reg Epoch {}/{}'.format(epoch, num_epochs))
#                 print('-' * 10)
#                 train_acc_history.append(epoch_acc)
#                 train_loss_history.append(epoch_loss)
#             else:
#                 val_acc_history.append(epoch_acc)
#                 val_loss_history.append(epoch_loss)
#
#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
#
#         if goal_failed or tiny_lr:
#             print('Break','\n','-' * 10)
#             break
#
#     result_dict = {'model': checkpoint['model_state_dict'],
#                    'train_loss_history': train_loss_history,
#                    'val_loss_history': val_loss_history,
#                    'train_acc_history': train_acc_history,
#                    'val_acc_history': val_acc_history}
#
#     if goal_failed:
#         result_dict['msg'] = 'goal_failed'
#     else:
#         result_dict['msg'] = 'tiny learning rate'
#     if show:
#         plt.figure(figsize=(16, 14))
#         plt.subplot(2, 1, 1)
#         plt.plot(result_dict['train_loss_history'], '-o')
#         plt.plot(result_dict['val_loss_history'], '-o')
#         plt.legend(['train', 'val'], loc='lower right')
#         plt.xlabel('iteration', fontsize=20)
#         plt.ylabel('loss', fontsize=20)
#
#         plt.subplot(2, 1, 2)
#         plt.yticks([0.1, 0.3, 0.5, 0.7, 0.9])
#         plt.plot(result_dict['train_acc_history'], '-o')
#         plt.plot(result_dict['val_acc_history'], '-o')
#         plt.legend(['train', 'val'], loc='lower right')
#         plt.ylabel('accuracy', fontsize=20)
#         plt.figure(figsize=(4, 4))
#         plt.show()
#
#     return result_dict
