import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import regression_weight_tuning_EU_LG_UA as EU_LG_UA

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

def reg_model(model, criterion, dataloaders, dataset_sizes, device,PATH='../weights/reg_checkpoint.pt',
              num_epochs=10, lr_epsilon=1e-8, lgep=0.3, rs=0.001, show=False):

    def cal_reg_term(model, rs=0.001):
        layers = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
        m = layers[0].in_features
        p = layers[0].out_features
        params = 0
        for p in model.parameters():
            params += torch.sum(p ** 2)
        reg_term = (rs / (p + 1 + p * (m + 1))) * params
        return reg_term

    def predict(outputs, lgep, labels, device):
        pred = torch.zeros(outputs.shape[0]).to(device)
        for i in range(outputs.shape[0]):
            if np.abs(outputs[i].item() - labels[i].item()) <= lgep:
                pred[i] = 1.0
            else:
                pred[i] = 0
        return pred

    # local variables
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    train_acc_history = []
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []
    lr_epsilon = lr_epsilon
    goal_failed = False
    tiny_lr = False
    checkpoint = None
    epoch = -1

    while epoch < num_epochs:
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, PATH)
        checkpoint = torch.load(PATH)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            running_loss = 0.0
            corrects = 0
            if phase == 'train':
                model.train()
            else:
                model.eval()
            if epoch == -1:
                for inputs, labels in dataloaders[phase]:
                    s = inputs.size(0)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        # forward
                        outputs = model(inputs)
                        reg_term = cal_reg_term(model, rs)
                        loss = criterion(outputs.squeeze(-1), labels) + reg_term
                        # statistics
                        running_loss += loss.item() * s
                        if phase == 'train':
                            # backward & adjust weights
                            loss.backward()
                            optimizer.step()
                epoch_loss = running_loss / dataset_sizes[phase]
                goal_achieved = learning_goal_1(model, dataloaders['train'], lgep, device)
                if goal_achieved:
                    former_loss = epoch_loss
                    epoch += 1
                    print('a')
                    break
                else:
                    goal_failed = True
                    break

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                s = inputs.size(0)
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # forward
                    outputs = model(inputs)
                    reg_term = cal_reg_term(model, rs)
                    loss = criterion(outputs.squeeze(-1), labels) + reg_term
                    pred = predict(outputs, lgep, labels, device)
                    # statistics
                    running_loss += loss.item() * s
                    corrects += torch.sum(pred).item()
                    if phase == 'train':
                        # backward & adjust weights
                        loss.backward()
                        optimizer.step()

            # statistics
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(corrects) / dataset_sizes[phase]

            if phase == 'train':
                # weight tuning
                if epoch_loss <= former_loss:
                    # learning goal
                    goal_achieved = learning_goal_1(model, dataloaders['train'], lgep, device)
                    if goal_achieved:
                        optimizer.param_groups[0]['lr'] *= 1.2
                        torch.save({'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict()}, PATH)
                        checkpoint = torch.load(PATH)
                        former_loss = epoch_loss
                    else:
                        goal_failed = True
                else:
                    if optimizer.param_groups[0]['lr'] < lr_epsilon:
                        tiny_lr = True
                    else:
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model.train()
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        optimizer.param_groups[0]['lr'] *= 0.7
                        torch.save({'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict()}, PATH)
                        checkpoint = torch.load(PATH)
                        break
                epoch += 1
                print('Reg Epoch {}/{}'.format(epoch, num_epochs))
                print('-' * 10)
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
            else:
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        if goal_failed or tiny_lr:
            print('Break','\n','-' * 10)
            break

    result_dict = {'model': checkpoint['model_state_dict'],
                   'train_loss_history': train_loss_history,
                   'val_loss_history': val_loss_history,
                   'train_acc_history': train_acc_history,
                   'val_acc_history': val_acc_history}

    if goal_failed:
        result_dict['msg'] = 'goal_failed'
    else:
        result_dict['msg'] = 'tiny learning rate'
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


# reorganize function below include regularization, weight-tunning and prunning hidden node
def reorg_model(model, criterion, dataloaders, dataset_sizes, device,PATH='../weights/reorg_checkpoint.pt',
                lr_epsilon=1e-6, lgep=0.25, n=1,rs=0.001):

    p = model.state_dict()['l1.weight'].shape[0]
    print(f'Init p: {p}')
    k = 0
    while k < p:
        result = reg_model(model, criterion, dataloaders, dataset_sizes, device, PATH='../weights/reg_checkpoint.pt',
                           num_epochs=20, lr_epsilon=lr_epsilon, lgep=lgep, rs=rs)

        # store model state dict
        torch.save(result['model'], PATH)
        model.load_state_dict(torch.load(PATH))
        msd = model.state_dict()

        # ignore some weights
        for key in msd.keys():
            if key == 'l1.weight':
                msd[key] = msd[key][[x for i, x in enumerate(range(p)) if i != k], :]
            elif key == 'l1.bias':
                msd[key] = msd[key][[x for i, x in enumerate(range(p)) if i != k]]
            elif key == 'l2.weight':
                msd[key] = msd[key][:, [x for i, x in enumerate(range(p)) if i != k]]

        m = next(iter(dataloaders['train']))[0].shape[-1]
        model = slfn(m, p).to(device)
        model.load_state_dict(msd)

        print('Start weight-tuning')
        result = EU_LG_UA.train_model_lgt1_reg(model, criterion, dataloaders, dataset_sizes, device,
                                                 PATH='../weights/train_checkpoint.pt',
                                                 epsilon=1e-6, num_epochs=50, lgep=0.3, show=False)
        print('Finish weight-tuning')
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

    return model.load_state_dict(torch.load(PATH)), p