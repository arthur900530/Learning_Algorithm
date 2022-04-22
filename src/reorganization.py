import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import regression_weight_tuning_EU_LG_UA as EU_LG_UA


def learning_goal_lsc(model, dataloader, device, n):
    clone_model = copy.deepcopy(model)
    one_min = np.ones(n) * 10
    zero_max = np.zeros(n) * (-10)
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = clone_model(inputs).squeeze(-1)
            outputs = outputs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            for i in range(outputs.shape[0]):
                if labels[i] == 1.0:
                    o = outputs[i]
                    one_min = np.append(one_min, o)
                    a = np.argmax(one_min)
                    one_min = np.delete(one_min, a)
                else:
                    o = np.array([outputs[i]])
                    zero_max = np.append(zero_max, o)
                    a = np.argmin(zero_max)
                    zero_max = np.delete(zero_max, a)
        om = np.max(one_min)
        zm = np.min(zero_max)
        v = (om + zm) / 2
        if om > zm:
            return True, v
        else:
            return False, v


def reg_model(model, criterion, optimizer, dataloaders, dataset_sizes, device,
              PATH='../weights/reg_checkpoint.pt', num_epochs=10,
              lr_epsilon=1e-8, n=10, rs=0.001, show=False):
    # local variables
    train_acc_history = []
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []
    lr_epsilon = lr_epsilon
    goal_failed = False
    tiny_lr = False
    checkpoint = None
    epoch = -1
    v = 0.7

    def cal_reg_term(model, rs=0.001):
        layers = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
        m = layers[0].in_features
        p = layers[0].out_features
        params = 0
        for p in model.parameters():
            params += torch.sum(p ** 2)
        reg_term = (rs / (p + 1 + p * (m + 1))) * params
        return reg_term

    def predict(outputs, v, device):
        pred = torch.zeros(outputs.shape[0]).to(device)
        for i in range(outputs.shape[0]):
            if outputs[i].item() < v:
                pred[i] = 0
            else:
                pred[i] = 1.0
        return pred

    def first_forward():
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
                reg_term = cal_reg_term(model, rs)
                loss = criterion(outputs.squeeze(-1), labels) + reg_term
                pred = predict(outputs, v, device)
                # statistics
                running_loss += loss.item() * s
                corrects += torch.sum(pred == labels.flatten().data)
                if phase == 'train':
                    # backward & adjust weights
                    loss.backward()
                    optimizer.step()
        epoch_loss = running_loss / dataset_sizes[phase]
        return model, epoch_loss

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
                model, epoch_loss = first_forward()
                goal_achieved, v = learning_goal_lsc(model, dataloaders[phase], device, n)
                if goal_achieved:
                    former_loss = epoch_loss
                    epoch += 1
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
                    pred = predict(outputs, v, device)
                    # statistics
                    running_loss += loss.item() * s
                    corrects += torch.sum(pred == labels.flatten().data)
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
                    goal_achieved, v = learning_goal_lsc(model, dataloaders[phase], device, n)
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
                print('Epoch {}/{}'.format(epoch, num_epochs))
                print('-' * 10)
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
            else:
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print(goal_failed, tiny_lr)
        if goal_failed or tiny_lr:
            print('Break')
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


def reorg_model(model, criterion, optimizer, dataloaders, dataset_sizes, device,
                PATH='../weights/reorg_checkpoint.pt', lr_epsilon=1e-8, n=10,rs=0.001):
    layers = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
    p = layers[0].out_features
    k = 0
    while k < p:
        result = reg_model(model, criterion, optimizer,
                           dataloaders, dataset_sizes,
                           device, num_epochs=10, lr_epsilon=lr_epsilon, n=n, rs=rs)

        # store model state dict
        torch.save(result['model'], PATH)
        model.load_state_dict(torch.load(PATH))
        msd = model.state_dict()
        for key in msd.keys():
            if key == '0.weight':
                msd[key] = msd[key][[x for i, x in enumerate(range(p)) if i != k], :]
            elif key == '0.bias':
                msd[key] = msd[key][[x for i, x in enumerate(range(p)) if i != k]]
            elif key == '2.weight':
                msd[key] = msd[key][:, [x for i, x in enumerate(range(p)) if i != k]]

        model = nn.Sequential(nn.Linear(12, p - 1), nn.ReLU(),
                              nn.Linear(p - 1, 1), nn.ReLU()).to(device)
        model.load_state_dict(msd)
        print('start weight-tuning')
        result = EU_LG_UA.train_model(model, criterion, optimizer,
                                      dataloaders, dataset_sizes, device,
                                      epsilon=lr_epsilon, num_epochs=50, n=n, show=False)
        if result['result']:
            print('p--')
            p -= 1
            torch.save(model.state_dict(), PATH)
        else:
            print('k++')
            k += 1
            model = nn.Sequential(nn.Linear(12, p), nn.ReLU(),
                                  nn.Linear(p, 1), nn.ReLU()).to(device)
            model.load_state_dict(torch.load(PATH))
        print(f'k: {k}, p: {p}')

    return model.load_state_dict(torch.load(PATH)), p