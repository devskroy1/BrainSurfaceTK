import os.path as osp

PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', '..')
import sys

sys.path.append(PATH_TO_ROOT)

import os
import time
import pickle
import csv

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from models.pointnet.src.models.pointnet2_regression import Net
from models.pointnet.src.utils import get_data_path, data


def train(model, train_loader, epoch, device, optimizer, scheduler, writer):
    model.train()
    loss_train = 0.0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Shape: B
        batch_tensor = torch.tensor(data.batch)
        # Shape: N x 3
        pos_tensor = torch.tensor(data.pos)
        # Shape: N x d_in
        x_tensor = torch.tensor(data.x)
        # Shape: B
        y_tensor = torch.tensor(data.y)

        # print("Inside pointnet2_segmentn train()")
        # Sometimes have inconsistencies in num_points, with same batch size. With batch size 2,
        # sometimes it is 10002, sometimes 10003.
        # When 10003, it leads to error: "RuntimeError: shape '[2, 5001, 3]' is invalid for input of size 30009"
        # Have resolved this by slicing closest multiple of batch size to current num points elements from tensors
        # print("batch_tensor shape")
        # print(batch_tensor.shape)
        # print("pos_tensor.shape")
        # print(pos_tensor.shape)
        # print("x_tensor shape")
        # print(x_tensor.shape)
        # print("y_tensor shape")
        # print(y_tensor.shape)

        num_points = pos_tensor.size(0)
        d_in = x_tensor.size(1)
        batch_size = torch.max(batch_tensor).item() + 1

        # print("batch size")
        # print(batch_size)

        quot = num_points // batch_size
        num_points_multiple = quot * batch_size

        pos_tensor_slice = pos_tensor[:num_points_multiple, :]
        x_tensor_slice = x_tensor[:num_points_multiple, :]
        batch_tensor_slice = batch_tensor[:num_points_multiple]
        y_tensor_slice = y_tensor[:num_points_multiple]

        pos_tensor = pos_tensor_slice.reshape(batch_size, quot, 3)
        x_tensor = x_tensor_slice.reshape(batch_size, quot, d_in)

        # pos_tensor = pos_tensor.reshape(batch_size, num_points // batch_size, 3)
        # x_tensor = x_tensor.reshape(batch_size, num_points // batch_size, d_in)

        pos_feature_data = torch.cat([pos_tensor, x_tensor], dim=2)
        pos_feature_data_float = torch.tensor(pos_feature_data, dtype=torch.float32)

        #pred = model(pos_feature_data_float)
        # print("Inside train()")
        # print("pred")
        # print(pred)
        # print("pred shape")
        # print(pred.shape)
        # print("y_tensor_slice float() shape")
        # print(y_tensor_slice.float().shape)
        # print("y_tensor_slice[:, 0].float() shape")
        # print(y_tensor_slice[:, 0].float().shape)

        pred = model(data)
        #loss = F.mse_loss(pred, y_tensor_slice[:, 0].float())

        print("Inside train()")
        print("pred")
        print(pred)
        print("pred shape")
        print(pred.shape)
        print("data.y[:, 0].float() shape")
        print(data.y[:, 0].float().shape)

        loss = F.mse_loss(pred, data.y[:, 0].float())

        loss.backward()
        optimizer.step()

        loss_train += loss.item()

    scheduler.step()

    if writer is not None:
        writer.add_scalar('Loss/train_mse', loss_train / len(train_loader), epoch)


def test_regression(model, loader, indices, device, recording, results_folder, val=True, epoch=0):
    model.eval()
    if recording:

        with open(results_folder + '/results.csv', 'a', newline='') as results_file:
            result_writer = csv.writer(results_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            if val:
                print('Validation'.center(60, '-'))
                result_writer.writerow(['Val scores Epoch - ' + str(epoch)])
            else:
                print('Test'.center(60, '-'))
                result_writer.writerow(['Test scores'])

            mse = 0
            l1 = 0
            for idx, data in enumerate(loader):

                data = data.to(device)
                with torch.no_grad():

                    # Shape: B
                    batch_tensor = torch.tensor(data.batch)
                    # Shape: N x 3
                    pos_tensor = torch.tensor(data.pos)
                    # Shape: N x d_in
                    x_tensor = torch.tensor(data.x)
                    # Shape: B
                    y_tensor = torch.tensor(data.y)

                    # print("Inside pointnet2_segmentn train()")
                    # Sometimes have inconsistencies in num_points, with same batch size. With batch size 2,
                    # sometimes it is 10002, sometimes 10003.
                    # When 10003, it leads to error: "RuntimeError: shape '[2, 5001, 3]' is invalid for input of size 30009"
                    # Have resolved this by slicing closest multiple of batch size to current num points elements from tensors
                    # print("batch_tensor shape")
                    # print(batch_tensor.shape)
                    # print("pos_tensor.shape")
                    # print(pos_tensor.shape)
                    # print("x_tensor shape")
                    # print(x_tensor.shape)
                    # print("y_tensor shape")
                    # print(y_tensor.shape)

                    num_points = pos_tensor.size(0)
                    d_in = x_tensor.size(1)
                    batch_size = torch.max(batch_tensor).item() + 1

                    # print("batch size")
                    # print(batch_size)

                    quot = num_points // batch_size
                    num_points_multiple = quot * batch_size

                    pos_tensor_slice = pos_tensor[:num_points_multiple, :]
                    x_tensor_slice = x_tensor[:num_points_multiple, :]
                    batch_tensor_slice = batch_tensor[:num_points_multiple]
                    y_tensor_slice = y_tensor[:num_points_multiple]

                    pos_tensor = pos_tensor_slice.reshape(batch_size, quot, 3)
                    x_tensor = x_tensor_slice.reshape(batch_size, quot, d_in)

                    # pos_tensor = pos_tensor.reshape(batch_size, num_points // batch_size, 3)
                    # x_tensor = x_tensor.reshape(batch_size, num_points // batch_size, d_in)

                    pos_feature_data = torch.cat([pos_tensor, x_tensor], dim=2)

                    pred = model(pos_feature_data)
                    print("Inside test_regression()")
                    print("pred shape")
                    print(pred.shape)
                    print("len(pred)")
                    print(len(pred))
                    #pred = model(data)
                    for i in range(len(pred)):
                        # print(str(pred[i].item()).center(20, ' '),
                        #       str(data.y[:, 0][i].item()).center(20, ' '),
                        #       indices[idx * len(pred) + i])
                        #
                        # result_writer.writerow([indices[idx * len(pred) + i][:11], indices[idx * len(pred) + i][12:],
                        #                         str(pred[i].item()), str(data.y[:, 0][i].item()),
                        #                         str(abs(pred[i].item() - data.y[:, 0][i].item()))])

                        print(str(pred[i].item()).center(20, ' '),
                              str(y_tensor_slice[:, 0][i].item()).center(20, ' '),
                              indices[idx * len(pred) + i])

                        result_writer.writerow([indices[idx * len(pred) + i][:11], indices[idx * len(pred) + i][12:],
                                                str(pred[i].item()), str(y_tensor_slice[:, 0][i].item()),
                                                str(abs(pred[i].item() - y_tensor_slice[:, 0][i].item()))])

                    loss_test_mse = F.mse_loss(pred, y_tensor_slice[:, 0])
                    loss_test_l1 = F.l1_loss(pred, y_tensor_slice[:, 0])
                    # loss_test_mse = F.mse_loss(pred, data.y[:, 0])
                    # loss_test_l1 = F.l1_loss(pred, data.y[:, 0])

                    mse += loss_test_mse.item()
                    l1 += loss_test_l1.item()
            if val:
                result_writer.writerow(['Epoch average error:', str(l1 / len(loader))])
                print(f'Epoch {epoch} average error: {l1 / len(loader)}')
            else:
                result_writer.writerow(['Test average error:', str(l1 / len(loader))])
                print(f'Test average error: {l1 / len(loader)}')
    else:

        if val:
            print('Validation'.center(60, '-'))
        else:
            print('Test'.center(60, '-'))

        mse = 0
        l1 = 0
        for idx, data in enumerate(loader):
            data = data.to(device)
            with torch.no_grad():

                # Shape: B
                batch_tensor = torch.tensor(data.batch)
                # Shape: N x 3
                pos_tensor = torch.tensor(data.pos)
                # Shape: N x d_in
                x_tensor = torch.tensor(data.x)
                # Shape: B
                y_tensor = torch.tensor(data.y)

                # print("Inside pointnet2_segmentn train()")
                # Sometimes have inconsistencies in num_points, with same batch size. With batch size 2,
                # sometimes it is 10002, sometimes 10003.
                # When 10003, it leads to error: "RuntimeError: shape '[2, 5001, 3]' is invalid for input of size 30009"
                # Have resolved this by slicing closest multiple of batch size to current num points elements from tensors
                # print("batch_tensor shape")
                # print(batch_tensor.shape)
                # print("pos_tensor.shape")
                # print(pos_tensor.shape)
                # print("x_tensor shape")
                # print(x_tensor.shape)
                # print("y_tensor shape")
                # print(y_tensor.shape)

                num_points = pos_tensor.size(0)
                d_in = x_tensor.size(1)
                batch_size = torch.max(batch_tensor).item() + 1

                # print("batch size")
                # print(batch_size)

                quot = num_points // batch_size
                num_points_multiple = quot * batch_size

                pos_tensor_slice = pos_tensor[:num_points_multiple, :]
                x_tensor_slice = x_tensor[:num_points_multiple, :]
                batch_tensor_slice = batch_tensor[:num_points_multiple]
                y_tensor_slice = y_tensor[:num_points_multiple]

                pos_tensor = pos_tensor_slice.reshape(batch_size, quot, 3)
                x_tensor = x_tensor_slice.reshape(batch_size, quot, d_in)

                # pos_tensor = pos_tensor.reshape(batch_size, num_points // batch_size, 3)
                # x_tensor = x_tensor.reshape(batch_size, num_points // batch_size, d_in)

                pos_feature_data = torch.cat([pos_tensor, x_tensor], dim=2)

                pred = model(pos_feature_data)

                #pred = model(data)

                for i in range(len(pred)):
                    # print(str(pred[i].item()).center(20, ' '),
                    #       str(data.y[:, 0][i].item()).center(20, ' '),
                    #       indices[idx * len(pred) + i])

                    print(str(pred[i].item()).center(20, ' '),
                          str(y_tensor_slice[:, 0][i].item()).center(20, ' '),
                          indices[idx * len(pred) + i])

                loss_test_mse = F.mse_loss(pred, y_tensor_slice[:, 0])
                loss_test_l1 = F.l1_loss(pred, y_tensor_slice[:, 0])
                # loss_test_mse = F.mse_loss(pred, data.y[:, 0])
                # loss_test_l1 = F.l1_loss(pred, data.y[:, 0])
                mse += loss_test_mse.item()
                l1 += loss_test_l1.item()

        if val:
            print(f'Epoch {epoch} average error (L1): {l1 / len(loader)}')
        else:
            print(f'Test average error (L1): {l1 / len(loader)}')

    return mse / len(loader), l1 / len(loader)


if __name__ == '__main__':

    PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..') + '/'

    num_workers = 2
    local_features = []
    global_features = []

    #################################################
    ########### EXPERIMENT DESCRIPTION ##############
    #################################################
    recording = False
    REPROCESS = False

    data_nativeness = 'native'
    data_compression = "10k"
    data_type = 'pial'
    hemisphere = 'both'

    comment = 'comment'

    #################################################
    ############ EXPERIMENT DESCRIPTION #############
    #################################################

    # 1. Model Parameters
    ################################################
    lr = 0.001
    batch_size = 2
    gamma = 0.9875
    scheduler_step_size = 2
    target_class = 'scan_age'
    task = 'regression'
    numb_epochs = 200
    number_of_points = 10000

    ################################################

    ########## INDICES FOR DATA SPLIT #############
    with open(PATH_TO_ROOT + 'src/names.pk', 'rb') as f:
        indices = pickle.load(f)
    ###############################################

    data_folder, files_ending = get_data_path(data_nativeness, data_compression, data_type, hemisphere=hemisphere)

    train_dataset, test_dataset, validation_dataset, train_loader, test_loader, val_loader, num_labels = data(
        data_folder,
        files_ending,
        data_type,
        target_class,
        task,
        REPROCESS,
        local_features,
        global_features,
        indices,
        batch_size,
        num_workers=2,
        data_nativeness=data_nativeness,
        data_compression=data_compression,
        hemisphere=hemisphere
    )

    if len(local_features) > 0:
        numb_local_features = train_dataset[0].x.size(1)
    else:
        numb_local_features = 0
    numb_global_features = len(global_features)

    # 7. Create the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(numb_local_features, numb_global_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=gamma)

    print(f'number of param: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    #################################################
    ############# EXPERIMENT LOGGING ################
    #################################################
    writer = None
    results_folder = None
    if recording:

        # Tensorboard writer.
        writer = SummaryWriter(log_dir='runs/' + task + '/' + comment, comment=comment)

        results_folder = 'runs/' + task + '/' + comment + '/results'
        model_dir = 'runs/' + task + '/' + comment + '/models'

        if not osp.exists(results_folder):
            os.makedirs(results_folder)

        if not osp.exists(model_dir):
            os.makedirs(model_dir)

        with open(results_folder + '/configuration.txt', 'w', newline='') as config_file:
            config_file.write('Learning rate - ' + str(lr) + '\n')
            config_file.write('Batch size - ' + str(batch_size) + '\n')
            config_file.write('Local features - ' + str(local_features) + '\n')
            config_file.write('Global feature - ' + str(global_features) + '\n')
            config_file.write('Number of points - ' + str(number_of_points) + '\n')
            config_file.write('Data res - ' + data_compression + '\n')
            config_file.write('Data type - ' + data_type + '\n')
            config_file.write('Data nativeness - ' + data_nativeness + '\n')
            # config_file.write('Additional comments - With rotate transforms' + '\n')

        with open(results_folder + '/results.csv', 'w', newline='') as results_file:
            result_writer = csv.writer(results_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            result_writer.writerow(['Patient ID', 'Session ID', 'Prediction', 'Label', 'Error'])

    #################################################
    #################################################

    best_val_loss = 999

    # MAIN TRAINING LOOP
    for epoch in range(1, numb_epochs + 1):
        start = time.time()
        train(model, train_loader, epoch, device,
              optimizer, scheduler, writer)

        val_mse, val_l1 = test_regression(model, val_loader,
                                          indices['Val'], device,
                                          recording, results_folder,
                                          epoch=epoch)

        if recording:
            writer.add_scalar('Loss/val_mse', val_mse, epoch)
            writer.add_scalar('Loss/val_l1', val_l1, epoch)

            end = time.time()
            print('Time: ' + str(end - start))
            if val_l1 < best_val_loss:
                best_val_loss = val_l1
                torch.save(model.state_dict(), model_dir + '/model_best.pt')
                print('Saving Model'.center(60, '-'))
            writer.add_scalar('Time/epoch', end - start, epoch)

    test_regression(model, test_loader, indices['Test'], device, recording, results_folder, val=False)

    if recording:
        # save the last model
        torch.save(model.state_dict(), model_dir + '/model_last.pt')

        # Eval best model on test
        model.load_state_dict(torch.load(model_dir + '/model_best.pt'))

        with open(results_folder + '/results.csv', 'a', newline='') as results_file:
            result_writer = csv.writer(results_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            result_writer.writerow(['Best model!'])

        test_regression(model, test_loader, indices['Test'], device, recording, results_folder, val=False)
