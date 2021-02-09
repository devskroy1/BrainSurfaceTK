import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import Module, Conv3d, ConvTranspose3d, Linear, ReLU, Sequential, Linear, Flatten, L1Loss, BatchNorm3d, \
    Dropout, BatchNorm1d
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from ..utils.utils import plot_preds
from ..utils.models import ImageSegmentationDataset, Part3, resample_image, PrintTensor
import os.path as osp
PATH_TO_VOLUME3D = osp.join(osp.dirname(osp.realpath(__file__)), '..') + '/'


def save_graphs_train(fn, num_epochs, training_loss, val_loss_epoch5):
    '''
    Saves all the necessary graphs
    :param fn: path to folder where to save
    :param num_epochs: epoch list
    :param training_loss: loss list
    :param test_loss_epoch5: test loss list
    :param writer: tensorboard writer
    :return:
    '''

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', f'{fn}/')


    plt.plot([epoch for epoch in range(num_epochs)], training_loss, color='b', label='Train')
    plt.plot([5 * i for i in range(len(val_loss_epoch5))], val_loss_epoch5, color='r', label='Val')
    plt.title("Loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.ylim(0, 5)
    plt.xlim(-5, num_epochs + 5)
    plt.legend()
    plt.savefig(path + f'graph.png')

    plt.close()

def save_graphs_train_test(fn, num_epochs, training_loss, test_loss_epoch5, writer):
    '''
    Saves all the necessary graphs
    :param fn: path to folder where to save
    :param num_epochs: epoch list
    :param training_loss: loss list
    :param test_loss_epoch5: test loss list
    :param writer: tensorboard writer
    :return:
    '''

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', f'{fn}/')

    plt.plot([epoch for epoch in range(num_epochs)], training_loss, color='b', label='Train')
    plt.plot([5*i for i in range(len(test_loss_epoch5))], test_loss_epoch5, color='r', label='Test')
    plt.title("Loss")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.ylim(0, 5)
    plt.xlim(-5, num_epochs+5)
    plt.legend()
    plt.savefig(path + f'graph.png')

    plt.close()

def save_to_log_train_val(model, params, fn, final_nll, num_epochs, batch_size, lr, feats, gamma, smoothen, edgen, dropout_p,
                img_spacing, img_size, scheduler_freq):
    '''
    Save all the information about the run to log
    '''

    print(f"Average NLL Loss on whole val set: {final_nll}")

    result = f"""
        ########################################################################

        *****    Score = {final_nll}    *****

        2. Number of epochs:
        num_epochs = {num_epochs}

        Batch size during training
        batch_size = {batch_size}

        Learning rate for optimizers
        lr = {lr}

        Size of feature amplifier
        Feature Amplifier: {feats}


        Gamma (using sched)
        Gamma: {gamma}
        Frequency of step: {scheduler_freq}

        7. Image spacing and size
        img_spacing = {img_spacing}
        img_size = {img_size}

        Smooth:
        smoothen = {smoothen}

        Edgen:
        edgen = {edgen}

        Amount of dropout:
        dropout_p = {dropout_p}

        Total number of parameters is: {params}

        Model:
        {model.__str__()}
        ########################################################################
        """

    with open(f'{fn}/log.txt', 'a+') as log:
        log.write('\n')
        log.write(result)
        log.write('\n')
        torch.save(model, f'{fn}/model.pth')

    path = osp.join(fn, '../')
    with open(path + 'all_log.txt', 'a+') as log:
        log.write('\n')
        log.write(f'SUBJECT #{fn[-1]}:    Validation = {final_nll},    ')


def save_to_log_test(model, params, fn, score, num_epochs, batch_size, lr, feats, gamma, smoothen, edgen, dropout_p,
                     img_spacing, img_size, scheduler_freq):
    '''
    Save all the information about the run to log
    '''

    print(f"Average nll Loss on whole test set: {score}")

    result = f"""
        ########################################################################

        *****   Score = {score}   *****

        2. Number of epochs:
        num_epochs = {num_epochs}

        3. Batch size during training
        batch_size = {batch_size}

        4. Learning rate for optimizers
        lr = {lr}

        5. Size of feature amplifier
        Feature Amplifier: {feats}

        6. Gamma (using sched)
        Gamma: {gamma}
        Frequency of step: {scheduler_freq}

        7. Image spacing and size
        img_spacing = {img_spacing}
        img_size = {img_size}

        7. Smooth:
        smoothen = {smoothen}

        8. Edgen:
        edgen = {edgen}

        9. Dropout:
        dropout_p = {dropout_p}

        Total number of parameters is: {params}

        # Model:
        {model.__str__()}
        ########################################################################
        """

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', f'{fn}/')

    with open(path + 'test_log.txt', 'a+') as log:
        log.write('\n')
        log.write(result)
        log.write('\n')
        torch.save(model, path + '/test_model.pth')

    path = osp.join(fn, '../')
    with open(path + 'all_log.txt', 'a+') as log:
        log.write(f'Test = {score} .')
        log.write('\n')

def train_val_classification(lr, feats, num_epochs, gamma, batch_size, dropout_p, dataset_train, dataset_val, fn, number_here, scheduler_freq, writer):
    # 1. Display GPU Settings:
    cuda_dev = '0'  # GPU device 0 (can be changed if multiple GPUs are available)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + cuda_dev if use_cuda else "cpu")
    print('Device: ' + str(device))
    if use_cuda:
        print('GPU: ' + str(torch.cuda.get_device_name(int(cuda_dev))))

    # 2. Define loss function
    loss_function = nll_loss()

    # 3. Print parameters
    print(f"Learning Rate: {lr} and Feature Amplifier: {feats}, Num_epochs: {num_epochs}, Gamma: {gamma}")

    # 4. Define collector lists
    folds_val_scores = []
    training_loss = []
    val_loss_epoch5 = []
    i_fold_val_scores = []

    # 5. Create data loaders
    train_loader = DataLoader(dataset_train, batch_size=batch_size)
    val_loader = DataLoader(dataset_val, batch_size=batch_size)

    # 6. Define a model
    model = Part3(feats, dropout_p).to(device=device)

    # 7. Print parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Params: {params}")

    # 8. Create an optimizer + LR scheduler
    optimizer = Adam(model.parameters(), lr, weight_decay=0.005)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma, last_epoch=-1)

    # 9. Proceed to train
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = []
        correct = 0
        for batch_data, batch_labels in train_loader:
            batch_labels = batch_labels.to(device=device)
            batch_data = batch_data.to(device=device)  # move to device, e.g. GPU
            batch_preds = model(batch_data)
            pred_label = pred.max(1)[1]
            loss = loss_function(batch_preds, batch_labels.y[:, 0].long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += pred_label.eq(batch_labels.y[:, 0].long()).sum().item()
            epoch_loss.append(loss.item())

        acc = correct / len(dataset_train)
        training_nll = np.mean(epoch_loss)
        training_loss.append(training_nll)

        writer.add_scalar('Acc/train', acc, epoch)
        writer.add_scalar('NLL Loss/train', training_nll, epoch)

        if epoch % scheduler_freq == 0:
            scheduler.step()

        # 10. Validate every N epochs

        if (epoch % 5 == 0):
            correct = 0
            val_loss = []
            model.eval()
            # pred_ages = []
            # actual_ages = []
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    batch_data = batch_data.to(device=device)  # move to device, e.g. GPU
                    batch_labels = batch_labels.to(device=device)
                    batch_preds = model(batch_data)
                    pred_label = batch_preds.max(1)[1]
                    correct += pred_label.eq(batch_labels.y[:, 0].long()).sum().item()
                    loss = loss_function(batch_preds, batch_labels.y[:, 0].long())
                    val_loss.append(loss.item())

                acc = correct / len(dataset_val)
                mean_val_error5 = np.mean(val_loss)
                val_loss_epoch5.append(mean_val_error5)

            print(f"Epoch: {epoch}:: Learning Rate: {scheduler.get_lr()[0]}")
            writer.add_scalar('Acc/val', acc, epoch)
            writer.add_scalar('NLL Loss/validate', mean_val_error5, epoch)

    # 11. Validate the last time
    correct = 0
    model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            batch_data = batch_data.to(device=device)  # move to device, e.g. GPU
            batch_labels = batch_labels.to(device=device)
            batch_preds = model(batch_data)
            pred_label = batch_preds.max(1)[1]
            correct += pred_label.eq(batch_labels.y[:, 0].long()).sum().item()
            loss = loss_function(batch_preds, batch_labels.y[:, 0].long())
            i_fold_val_scores.append(loss.item())

    # 12. Summarise the results
    acc = correct / len(dataset_val)
    mean_fold_score = np.mean(i_fold_val_scores)
    val_loss_epoch5.append(mean_fold_score)
    print(f"Mean Classification error: {mean_fold_score}")
    print(f"Final validation accuracy: {acc}")
    writer.add_scalar('Mean Classification error/val', mean_fold_score)
    writer.add_scalar('Final Acc/validate', acc)


    folds_val_scores.append(mean_fold_score)

    final_nll = np.mean(folds_val_scores)

    save_graphs_train(fn, num_epochs, training_loss, val_loss_epoch5)

    return model, params, final_nll

def train_test_classification(lr, feats, num_epochs, gamma, batch_size, dropout_p, dataset_train, dataset_test, fn, number_here, scheduler_freq, writer):
    '''
    Main train-test loop. Train on training data and evaluate on testing data.

    :param lr: learning rate
    :param feats: feature amplifier (multiplier of the number of parameters in the CNN)
    :param num_epochs:
    :param gamma: scheduler gamma
    :param batch_size
    :param dropout_p: dropout proba
    :param dataset_train
    :param dataset_test
    :param fn: saving folder
    :param scheduler_freq
    :param writer: tensorboard
    :return: model, params, score, train_loader, test_loader
    '''

    # 1. Display GPU Settings:
    cuda_dev = '0'  # GPU device 0 (can be changed if multiple GPUs are available)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + cuda_dev if use_cuda else "cpu")
    print('Device: ' + str(device))
    if use_cuda:
        print('GPU: ' + str(torch.cuda.get_device_name(int(cuda_dev))))

    # 2. Define loss function
    loss_function = nll_loss()

    # 3. Print parameters
    print(f"Learning Rate: {lr} and Feature Amplifier: {feats}, Num_epochs: {num_epochs}, Gamma: {gamma}")

    # 4. Define collector lists
    training_loss = []
    test_loss_epoch5 = []

    # 5. Create data loaders
    train_loader = DataLoader(dataset_train, batch_size=batch_size)
    test_loader = DataLoader(dataset_test, batch_size=batch_size)

    # 6. Define a model
    model = Part3(feats, dropout_p).to(device=device)

    # 7. Print parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Params: {params}")

    # 8. Create an optimizer + LR scheduler
    optimizer = Adam(model.parameters(), lr, weight_decay=0.005)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma, last_epoch=-1)

    # 9. Proceed to train
    for epoch in range(num_epochs):

        model.train()
        epoch_loss = []
        for batch_data, batch_labels in train_loader:
            batch_labels = batch_labels.to(device=device)
            batch_data = batch_data.to(device=device)  # move to device, e.g. GPU
            batch_preds = model(batch_data)
            loss = loss_function(batch_preds, batch_labels.y[:, 0].long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        training_nll = np.mean(epoch_loss)
        training_loss.append(training_nll)

        if epoch % scheduler_freq == 0:
            scheduler.step()

        # 10. Validate every N epochs
        if (epoch % 5 == 0):

            correct = 0
            test_loss = []
            model.eval()
            # pred_ages = []
            # actual_ages = []
            with torch.no_grad():
                for batch_data, batch_labels in test_loader:
                    batch_data = batch_data.to(device=device)  # move to device, e.g. GPU
                    batch_labels = batch_labels.to(device=device)
                    batch_preds = model(batch_data)
                    pred_label = batch_preds.max(1)[1]
                    correct += pred_label.eq(batch_labels.y[:, 0].long()).sum().item()
                    loss = loss_function(batch_preds, batch_labels.y[:, 0].long())
                    test_loss.append(loss.item())

                acc = correct / len(dataset_test)
                mean_test_error5 = np.mean(test_loss)
                test_loss_epoch5.append(mean_test_error5)

            print(f"Epoch: {epoch}:: Learning Rate: {scheduler.get_lr()[0]}")
            print(
                f"{number_here}:: Maxiumum Classification Error: {np.round(np.max(epoch_loss))} Average training classification Error: {training_nll}, NLL Test: {mean_test_error5}")

            writer.add_scalar('Max Classification Error/test', np.round(np.max(epoch_loss)), epoch)
            writer.add_scalar('NLL Loss/test', mean_test_error5, epoch)
            writer.add_scalar('Acc/test', acc, epoch)

    # 11. Validate the last time
    correct = 0
    model.eval()
    test_scores = []
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device=device)  # move to device, e.g. GPU
            batch_labels = batch_labels.to(device=device)
            batch_preds = model(batch_data)
            pred_label = batch_preds.max(1)[1]
            correct += pred_label.eq(batch_labels.y[:, 0].long()).sum().item()
            loss = loss_function(batch_preds, batch_labels.y[:, 0].long())
            test_scores.append(loss.item())

    # 12. Summarise the results
    acc = correct / len(dataset_test)
    score = np.mean(test_scores)
    test_loss_epoch5.append(score)

    print(f"Mean Classification Error: {score}")
    print(f"Final test accuracy: {acc}")
    writer.add_scalar('Mean Classification error/test', score)
    writer.add_scalar('Final Acc/test', acc)

    save_graphs_train_test(fn, num_epochs, training_loss, test_loss_epoch5, writer=writer)

    return model, params, score, train_loader, test_loader