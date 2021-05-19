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
#from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import DataLoader

from ..utils.utils import plot_preds
from ..utils.models import ImageSegmentationDataset, Part3, resample_image, PrintTensor
from ..utils.volumeCnnGCN import VolumeCNN_GCNRegressor
import os.path as osp

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


def save_to_log(model, params, fn, final_MAE, num_epochs, batch_size, lr, feats, gamma, smoothen, edgen, dropout_p, img_spacing, img_size, scheduler_freq):
    '''
    Save all the information about the run to log
    '''

    print(f"Average Loss on whole val set: {final_MAE}")

    result = f"""
        ########################################################################
        
        *****    Score = {final_MAE}    *****

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
        log.write(f'SUBJECT #{fn[-1]}:    Validation = {final_MAE},    ')

def denorm_target_f(target, dataset):
    # used to unstandardise the target (to get the scan age of the patient)
    return (target.cpu() * dataset.targets_std) + dataset.targets_mu

def find_subjects_data_volcnn(subjects, dataset_volcnn):

    volcnn_samples_shape = list(dataset_volcnn.samples[0].shape)
    batch_size_list = [len(subjects)]
    batch_data_shape = batch_size_list + volcnn_samples_shape
    batch_data = torch.empty(batch_data_shape, dtype=torch.float)
    batch_labels = torch.empty((len(subjects), 1), dtype=torch.float)

    # batch_data = []
    # batch_labels = []
    # print("len(subjects)")
    # print(len(subjects))
    # print("Inside find_subjects_data_volcnn()")
    # print("subjects")
    # print(subjects)
    # print("dataset_volcnn_train.ids")
    # print(dataset_volcnn_train.ids)
    for i in range(len(subjects)):
        subject = subjects[i][0]
        # print("subject")
        # print(subject)
        subject_id, sess_id = subject.split("_")
        for j in range(len(dataset_volcnn.ids)):
            volcnn_subject_id = dataset_volcnn.ids[j][0]
            volcnn_sess_id = dataset_volcnn.ids[j][1]
            if subject_id == volcnn_subject_id and sess_id == volcnn_sess_id:
                # print("subject_id")
                # print(subject_id)
                # print("sess_id")
                # print(sess_id)
                # print("dataset_volcnn_train.samples shape")
                # print(dataset_volcnn_train.samples.shape)
                # print("dataset_volcnn_train.samples")
                # print(dataset_volcnn_train.samples)
                # print("dataset_volcnn_train.samples[j].shape")
                # print(dataset_volcnn_train.samples[j].shape)
                batch_data[i] = dataset_volcnn.samples[j]
                batch_labels[i] = dataset_volcnn.targets[j]
                break
    # print("batch_Data")
    # print(batch_data)
    # print("batch_data")
    # print(batch_data)
    # print("batch_labels")
    # print(batch_labels)
    return batch_data, batch_labels

#Both Volume3DCNN and GCN
def train_validate(lr, feats, num_epochs, gamma, batch_size, dropout_p, dataset_volumecnn_train, dataset_volumecnn_val, fn, number_here,
                   scheduler_freq, writer, train_dl_gcn, val_dl_gcn, train_ds_gcn, val_ds_gcn):
    '''
    Main train-val loop. Train on training data and evaluate on validation data.

    :param lr: learning rate
    :param feats: feature amplifier (multiplier of the number of parameters in the CNN)
    :param num_epochs:
    :param gamma: scheduler gamma
    :param batch_size
    :param dropout_p: dropout proba
    :param dataset_train
    :param dataset_val
    :param fn: saving folder
    :param scheduler_freq
    :param writer: tensorboard
    :return: model, params, final_MAE
    '''

    # 1. Display GPU Settings:
    cuda_dev = '0'  # GPU device 0 (can be changed if multiple GPUs are available)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + cuda_dev if use_cuda else "cpu")
    print('Device: ' + str(device))
    if use_cuda:
        print('GPU: ' + str(torch.cuda.get_device_name(int(cuda_dev))))

    # 2. Define loss function
    loss_function = L1Loss()

    # 3. Print parameters
    print(f"Learning Rate: {lr} and Feature Amplifier: {feats}, Num_epochs: {num_epochs}, Gamma: {gamma}")

    # 4. Define collector lists
    folds_val_scores = []
    training_loss = []
    val_loss_epoch5 = []
    i_fold_val_scores = []

    # 5. Create data loaders
    train_loader_volcnn = DataLoader(dataset_volumecnn_train, batch_size=batch_size)
    val_loader_volcnn = DataLoader(dataset_volumecnn_val, batch_size=batch_size)

    # 6. Define a model
    #model = Part3(feats, dropout_p).to(device=device)
    in_dim = 6
    hidden_dim = 64
    model = VolumeCNN_GCNRegressor(feats, dropout_p, in_dim, hidden_dim, device).to(device)

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
        #for batch_data, batch_labels in train_loader_volcnn:
            # print("batch data")
            # print(batch_data)
        for iter, (subjects, bg, batch_labels_gcn) in enumerate(train_dl_gcn):
            # print("bg")
            # print(bg)
            # print("batch_labels_gcn shape")
            # print(batch_labels_gcn.shape)
            batch_data_volcnn, batch_labels_volcnn = find_subjects_data_volcnn(subjects, dataset_volumecnn_train)
            # print("batch_data_volcnn")
            # print(batch_data_volcnn)
            # print("batch_labels_volcnn")
            # print(batch_labels_volcnn)

            # print("batch_labels shape")
            # print(batch_labels.shape)
            bg = bg.to(device)
            bg_node_features = bg.ndata["features"].to(device)
            # print("bg_node_features")
            # print(bg_node_features)
           # bg = dgl.add_self_loop(bg)
            batch_data = batch_data_volcnn.to(device=device)  # move to device, e.g. GPU
            batch_labels_gcn = batch_labels_gcn.to(device)
            prediction = model(batch_data, bg, bg_node_features)

            # batch_data = batch_data.to(device=device)
            # prediction = model(batch_data)

            # print("pred shape")
            # print(prediction.shape)
            # print("prediction")
            # print(prediction)
            # #
            # print("batch_labels_gcn")
            # print(batch_labels_gcn)
            # print("batch_labels_gcn shape")
            # print(batch_labels_gcn.shape)
            #Should be 4 x 1
            # print(batch_labels_gcn)


            #loss = loss_function(prediction, batch_labels_gcn)
            predicted_age = denorm_target_f(prediction, train_ds_gcn)
            true_age = denorm_target_f(batch_labels_gcn, train_ds_gcn)
            # print("pred age")
            # print(predicted_age)
            # print("true age")
            # print(true_age)

            loss = loss_function(predicted_age, true_age)
            # print("loss")
            # print(loss)
            #loss = loss_function(prediction, batch_labels_volcnn)

            # batch_labels = batch_labels_volcnn.to(device=device)
            # batch_data = batch_data_volcnn.to(device=device)  # move to device, e.g. GPU
            # batch_preds = model(batch_data)
            # loss = loss_function(batch_preds, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        training_MAE = np.mean(epoch_loss)
        training_loss.append(training_MAE)

        writer.add_scalar('MAE Loss/train', training_MAE, epoch)

        if epoch % scheduler_freq == 0:
            scheduler.step()

        # 10. Validate every N epochs
        if (epoch % 5 == 0):
            val_loss = []
            model.eval()
            pred_ages = []
            actual_ages = []
            with torch.no_grad():
                #for batch_data, batch_labels in val_loader:
                for iter, (subjects, bg, batch_labels_gcn) in enumerate(val_dl_gcn):
                    batch_data_volcnn, batch_labels_volcnn = find_subjects_data_volcnn(subjects,
                                                                                       dataset_volumecnn_val)

                    # batch_data = batch_data_volcnn.to(device=device)  # move to device, e.g. GPU
                    # batch_labels = batch_labels_volcnn.to(device=device)
                    # batch_preds = model(batch_data)
                    #
                    # pred_ages.append([batch_preds[i].item() for i in range(len(batch_preds))])
                    # actual_ages.append([batch_labels[i].item() for i in range(len(batch_labels))])
                    #
                    # loss = loss_function(batch_preds, batch_labels)
                    # val_loss.append(loss.item())

                    bg = bg.to(device)
                    bg_node_features = bg.ndata["features"].to(device)
                    batch_data = batch_data_volcnn.to(device=device)  # move to device, e.g. GPU
                    batch_labels_gcn = batch_labels_gcn.to(device)
                    prediction = model(batch_data, bg, bg_node_features)

                    predicted_age = denorm_target_f(prediction, val_ds_gcn)
                    true_age = denorm_target_f(batch_labels_gcn, val_ds_gcn)

                    # pred_ages.append([prediction[i].item() for i in range(len(prediction))])
                    # actual_ages.append([batch_labels_gcn[i].item() for i in range(len(batch_labels_gcn))])
                    pred_ages.append([predicted_age[i].item() for i in range(len(predicted_age))])
                    actual_ages.append([true_age[i].item() for i in range(len(true_age))])

                    #loss = loss_function(prediction, batch_labels_gcn)
                    loss = loss_function(predicted_age, true_age)

                    val_loss.append(loss.item())

                mean_val_error5 = np.mean(val_loss)
                val_loss_epoch5.append(mean_val_error5)

            plot_preds(pred_ages, actual_ages, writer, epoch, test=False)

            print(f"Epoch: {epoch}:: Learning Rate: {scheduler.get_lr()[0]}")
            print(
                f"{number_here}:: Maxiumum Age Error: {np.round(np.max(epoch_loss))} Average Age Error: {training_MAE}, MAE Validation: {mean_val_error5}")

            writer.add_scalar('Max Age Error/validate', np.round(np.max(epoch_loss)), epoch)
            writer.add_scalar('MAE Loss/validate', mean_val_error5, epoch)

    # 11. Validate the last time
    model.eval()
    pred_ages = []
    actual_ages = []
    with torch.no_grad():
        #for batch_data, batch_labels in val_loader:
        for iter, (subjects, bg, batch_labels_gcn) in enumerate(val_dl_gcn):
            batch_data_volcnn, batch_labels_volcnn = find_subjects_data_volcnn(subjects, dataset_volumecnn_val)

            # batch_data = batch_data_volcnn.to(device=device)  # move to device, e.g. GPU
            # batch_labels = batch_labels_volcnn.to(device=device)
            # # batch_data = batch_data.to(device=device)  # move to device, e.g. GPU
            # # batch_labels = batch_labels.to(device=device)
            #
            # batch_preds = model(batch_data)
            #
            # pred_ages.append([batch_preds[i].item() for i in range(len(batch_preds))])
            # actual_ages.append([batch_labels[i].item() for i in range(len(batch_labels))])
            #
            # loss = loss_function(batch_preds, batch_labels)
            # i_fold_val_scores.append(loss.item())

            bg = bg.to(device)
            bg_node_features = bg.ndata["features"].to(device)
            batch_data = batch_data_volcnn.to(device=device)  # move to device, e.g. GPU
            batch_labels_gcn = batch_labels_gcn.to(device)
            prediction = model(batch_data, bg, bg_node_features)

            # pred_ages.append([prediction[i].item() for i in range(len(prediction))])
            # actual_ages.append([batch_labels_gcn[i].item() for i in range(len(batch_labels_gcn))])
            # loss = loss_function(prediction, batch_labels_gcn)

            predicted_age = denorm_target_f(prediction, val_ds_gcn)
            true_age = denorm_target_f(batch_labels_gcn, val_ds_gcn)

            # pred_ages.append([prediction[i].item() for i in range(len(prediction))])
            # actual_ages.append([batch_labels_gcn[i].item() for i in range(len(batch_labels_gcn))])
            pred_ages.append([predicted_age[i].item() for i in range(len(predicted_age))])
            actual_ages.append([true_age[i].item() for i in range(len(true_age))])

            loss = loss_function(predicted_age, true_age)

            i_fold_val_scores.append(loss.item())

    plot_preds(pred_ages, actual_ages, writer, epoch, test=False)

    # 12. Summarise the results
    mean_fold_score = np.mean(i_fold_val_scores)
    val_loss_epoch5.append(mean_fold_score)
    print(f"Mean Age Error: {mean_fold_score}")

    folds_val_scores.append(mean_fold_score)

    final_MAE = np.mean(folds_val_scores)

    save_graphs_train(fn, num_epochs, training_loss, val_loss_epoch5)

    return model, params, final_MAE

#Just Volume3DCNN
# def train_validate(lr, feats, num_epochs, gamma, batch_size, dropout_p, dataset_train, dataset_val, fn, number_here, scheduler_freq, writer):
#     '''
#     Main train-val loop. Train on training data and evaluate on validation data.
#
#     :param lr: learning rate
#     :param feats: feature amplifier (multiplier of the number of parameters in the CNN)
#     :param num_epochs:
#     :param gamma: scheduler gamma
#     :param batch_size
#     :param dropout_p: dropout proba
#     :param dataset_train
#     :param dataset_val
#     :param fn: saving folder
#     :param scheduler_freq
#     :param writer: tensorboard
#     :return: model, params, final_MAE
#     '''
#
#     # 1. Display GPU Settings:
#     cuda_dev = '0'  # GPU device 0 (can be changed if multiple GPUs are available)
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda:" + cuda_dev if use_cuda else "cpu")
#     print('Device: ' + str(device))
#     if use_cuda:
#         print('GPU: ' + str(torch.cuda.get_device_name(int(cuda_dev))))
#
#     # 2. Define loss function
#     loss_function = L1Loss()
#
#     # 3. Print parameters
#     print(f"Learning Rate: {lr} and Feature Amplifier: {feats}, Num_epochs: {num_epochs}, Gamma: {gamma}")
#
#     # 4. Define collector lists
#     folds_val_scores = []
#     training_loss = []
#     val_loss_epoch5 = []
#     i_fold_val_scores = []
#
#     # 5. Create data loaders
#     train_loader = DataLoader(dataset_train, batch_size=batch_size)
#     val_loader = DataLoader(dataset_val, batch_size=batch_size)
#
#     print("dataset_train.samples")
#     print(dataset_train.samples)
#
#     # 6. Define a model
#     model = Part3(feats, dropout_p).to(device=device)
#
#     # 7. Print parameters
#     params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Total Params: {params}")
#
#     # 8. Create an optimizer + LR scheduler
#     optimizer = Adam(model.parameters(), lr, weight_decay=0.005)
#     scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma, last_epoch=-1)
#
#     # 9. Proceed to train
#     for epoch in range(num_epochs):
#         model.train()
#         epoch_loss = []
#         for batch_data, batch_labels in train_loader:
#
#             # print("batch_data")
#             # print(batch_data)
#             # print("batch labels")
#             # print(batch_labels)
#             batch_labels = batch_labels.to(device=device)
#             batch_data = batch_data.to(device=device)  # move to device, e.g. GPU
#             batch_preds = model(batch_data)
#             loss = loss_function(batch_preds, batch_labels)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             epoch_loss.append(loss.item())
#
#
#         training_MAE = np.mean(epoch_loss)
#         training_loss.append(training_MAE)
#
#         writer.add_scalar('MAE Loss/train', training_MAE, epoch)
#
#
#         if epoch % scheduler_freq == 0:
#             scheduler.step()
#
#         # 10. Validate every N epochs
#         if (epoch % 5 == 0):
#             val_loss = []
#             model.eval()
#             pred_ages = []
#             actual_ages = []
#             with torch.no_grad():
#                 for batch_data, batch_labels in val_loader:
#                     batch_data = batch_data.to(device=device)  # move to device, e.g. GPU
#                     batch_labels = batch_labels.to(device=device)
#                     batch_preds = model(batch_data)
#
#                     pred_ages.append([batch_preds[i].item() for i in range(len(batch_preds))])
#                     actual_ages.append([batch_labels[i].item() for i in range(len(batch_labels))])
#
#                     loss = loss_function(batch_preds, batch_labels)
#                     val_loss.append(loss.item())
#
#                 mean_val_error5 = np.mean(val_loss)
#                 val_loss_epoch5.append(mean_val_error5)
#
#             plot_preds(pred_ages, actual_ages, writer, epoch, test=False)
#
#             print(f"Epoch: {epoch}:: Learning Rate: {scheduler.get_lr()[0]}")
#             print(
#                 f"{number_here}:: Maxiumum Age Error: {np.round(np.max(epoch_loss))} Average Age Error: {training_MAE}, MAE Validation: {mean_val_error5}")
#
#             writer.add_scalar('Max Age Error/validate', np.round(np.max(epoch_loss)), epoch)
#             writer.add_scalar('MAE Loss/validate', mean_val_error5, epoch)
#
#
#     # 11. Validate the last time
#     model.eval()
#     pred_ages = []
#     actual_ages = []
#     with torch.no_grad():
#         for batch_data, batch_labels in val_loader:
#             batch_data = batch_data.to(device=device)  # move to device, e.g. GPU
#             batch_labels = batch_labels.to(device=device)
#             batch_preds = model(batch_data)
#
#             pred_ages.append([batch_preds[i].item() for i in range(len(batch_preds))])
#             actual_ages.append([batch_labels[i].item() for i in range(len(batch_labels))])
#
#             loss = loss_function(batch_preds, batch_labels)
#             i_fold_val_scores.append(loss.item())
#
#     plot_preds(pred_ages, actual_ages, writer, epoch, test=False)
#
#     # 12. Summarise the results
#     mean_fold_score = np.mean(i_fold_val_scores)
#     val_loss_epoch5.append(mean_fold_score)
#     print(f"Mean Age Error: {mean_fold_score}")
#
#     folds_val_scores.append(mean_fold_score)
#
#     final_MAE = np.mean(folds_val_scores)
#
#     save_graphs_train(fn, num_epochs, training_loss, val_loss_epoch5)
#
#     return model, params, final_MAE