import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


def learning_goal_lsc(model, dataloader, v, device, n):
    clone_model = copy.deepcopy(model)
    q_one_min = 1
    q_zero_max = 0
    one_min = np.ones(n)
    zero_max = np.zeros(n)
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
                    if o > v and o < q_one_min:
                        q_one_min = o
                    one_min = np.append(one_min, o)
                    a = np.argmax(one_min)
                    one_min = np.delete(one_min, a)
                else:
                    o = np.array([outputs[i]])
                    if o < v and o > q_zero_max:
                        q_zero_max = o
                    zero_max = np.append(zero_max, o)
                    a = np.argmin(zero_max)
                    zero_max = np.delete(zero_max, a)
    omin = np.max(one_min)
    zmax = np.min(zero_max)

    if q_zero_max != 0 and q_one_min != 1:
        v = (q_one_min + q_zero_max)/2
        if omin > zmax :
            return True, v, q_one_min, q_zero_max
        else:
            return False, v, q_one_min, q_zero_max
    else:
        if omin > zmax :
            return True, v, v + 0.1, v - 0.1
        else:
            return False, v, v + 0.1, v - 0.1


def train_model(model, criterion, dataloaders, dataset_sizes, device, PATH = '../weights/train_checkpoint.pt', epsilon=1e-6, num_epochs=30, n=1, show=True, v=0.6):
    def predict(outputs, v, device):
      pred = torch.zeros(outputs.shape[0]).to(device)
      for i in range(outputs.shape[0]):
        if outputs[i].item() < v:
          pred[i] = 0
        else:
          pred[i] = 1.0
      return pred

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    train_acc_history = []
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []
    lr_epsilon = epsilon
    goal_achieved = False
    tiny_lr = False
    checkpoint = None
    epoch = 0
    v = v

    while epoch < num_epochs:
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
                # learning goal
                goal_achieved, v, omin, zmax  = learning_goal_lsc(model, dataloaders[phase], v, device, n)
                if goal_achieved:
                    break
                # weight tuning
                if epoch == 0:
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'former_loss': epoch_loss}, PATH)
                    checkpoint = torch.load(PATH)

                elif epoch_loss < checkpoint['former_loss']:
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
                # print('Epoch {}/{}'.format(epoch, num_epochs))
                # print('-' * 10)
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
            else:
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        if goal_achieved or tiny_lr:
          break


    result_dict = {'train_loss_history':train_loss_history,
                   'val_loss_history':val_loss_history,
                   'train_acc_history':train_acc_history,
                   'val_acc_history':val_acc_history,
                   'v':v, 'one_min':omin, 'zero_max':zmax}

    if goal_achieved:
        result_dict['result'] = True
        result_dict['msg'] = 'goal_achieved'
    elif tiny_lr:
        result_dict['result'] = False
        result_dict['msg'] = 'tiny learning rate'
    else:
        result_dict['result'] = False
        result_dict['msg'] = 'trained all epoch'
    if show:  
      plt.figure(figsize=(16,14))
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
      plt.figure(figsize=(4,4))
      plt.show()

    return result_dict

