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

from models.pointnet.src.models.pointnet2_classification import Net
from models.pointnet.src.utils import get_data_path, data


def train(model, train_loader, epoch, device, optimizer, scheduler, writer):
    model.train()
    loss_train = 0.0
    correct = 0

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for data in train_loader:
        print("data.y size")
        print(data.y.size())
        print("data.y")
        print(data.y)
        data = data.to(device)
        pred = model(data)
        perd_label = pred.max(1)[1]
        #loss = F.nll_loss(pred, data.y[:, 0].long())
        loss = F.smooth_l1_loss(pred, data.y[:, 0].long())
        loss.backward()
        optimizer.step()
        correct += perd_label.eq(data.y[:, 0].long()).sum().item()
        if (perd_label > 0) and (data.y[:, 0].long() > 0):
            true_positives += 1
        elif (perd_label > 0) and (data.y[:, 0].long() == 0):
            false_positives += 1
        elif (perd_label == 0) and (data.y[:, 0].long() > 0):
            false_negatives += 1
        loss_train += loss.item()
    acc = correct / len(train_loader.dataset)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = (2 * precision * recall) / (precision + recall)
    scheduler.step()

    if writer is not None:
        writer.add_scalar('Acc/train', acc, epoch)
        writer.add_scalar('Precision/train', precision, epoch)
        writer.add_scalar('Recall/train', recall, epoch)
        writer.add_scalar('F1 Score/train', f1_score, epoch)
        writer.add_scalar('Loss/train_huber', loss_train / len(train_loader), epoch)
    print('Train acc: ' + str(acc))
    print('Train precision: ' + str(precision))
    print('Train recall: ' + str(recall))
    print('Train F1 Score: ' + str(f1_score))
    print('Loss/train_huber: ' + str(loss_train / len(train_loader)))

def test_classification(model, loader, indices, device, recording, results_folder, val=True, epoch=0):
    model.eval()

    if recording:

        with open(results_folder + '/results.csv', 'a', newline='') as results_file:
            result_writer = csv.writer(results_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            if val:
                print('Validation'.center(60, '-'))
                result_writer.writerow(['Val accuracy Epoch - ' + str(epoch)])
                result_writer.writerow(['Val precision Epoch - ' + str(epoch)])
                result_writer.writerow(['Val recall Epoch - ' + str(epoch)])
                result_writer.writerow(['Val f1 score Epoch - ' + str(epoch)])
            else:
                print('Test'.center(60, '-'))
                result_writer.writerow(['Test accuracy'])
                result_writer.writerow(['Test precision'])
                result_writer.writerow(['Test recall'])
                result_writer.writerow(['Test f1 score'])

            correct = 0
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            for idx, data in enumerate(loader):
                data = data.to(device)
                with torch.no_grad():
                    pred = model(data).max(1)[1]

                    for i in range(len(pred)):
                        print(str(pred[i].item()).center(20, ' '),
                              str(data.y[:, 0][i].item()).center(20, ' '),
                              indices[idx * len(pred) + i])

                        result_writer.writerow([indices[idx * len(pred) + i][:11], indices[idx * len(pred) + i][12:],
                                                str(pred[i].item()), str(data.y[:, 0][i].item()),
                                                str(abs(pred[i].item() - data.y[:, 0][i].item()))])

                correct += pred.eq(data.y[:, 0].long()).sum().item()
                if (perd_label > 0) and (data.y[:, 0].long() > 0):
                    true_positives += 1
                elif (perd_label > 0) and (data.y[:, 0].long() == 0):
                    false_positives += 1
                elif (perd_label == 0) and (data.y[:, 0].long() > 0):
                    false_negatives += 1

            acc = correct / len(loader.dataset)
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f1_score = (2 * precision * recall) / (precision + recall)

            if val:
                print(f'Epoch {epoch} validation accuracy: {acc}')
                result_writer.writerow(['Epoch average val accuracy:', str(acc)])
                print(f'Epoch {epoch} validation precision: {precision}')
                result_writer.writerow(['Epoch validation precision:', str(precision)])
                print(f'Epoch {epoch} validation recall: {recall}')
                result_writer.writerow(['Epoch validation recall:', str(recall)])
                print(f'Epoch {epoch} validation F1 score: {f1_score}')
                result_writer.writerow(['Epoch validation f1 score:', str(f1_score)])

            else:
                print(f'Test accuracy: {acc}')
                result_writer.writerow(['Test average accuracy:', str(acc)])
                print(f'Test precision: {precision}')
                result_writer.writerow(['Test precision:', str(precision)])
                print(f'Test recall: {recall}')
                result_writer.writerow(['Test recall:', str(recall)])
                print(f'Test F1 score: {f1_score}')
                result_writer.writerow(['Test F1 score:', str(f1_score)])
    else:
        if val:
            print('Validation'.center(60, '-'))
        else:
            print('Test'.center(60, '-'))

        correct = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for idx, data in enumerate(loader):
            data = data.to(device)
            with torch.no_grad():
                pred = model(data).max(1)[1]

                for i in range(len(pred)):
                    print(str(pred[i].item()).center(20, ' '),
                          str(data.y[:, 0][i].item()).center(20, ' '),
                          indices[idx * len(pred) + i])

            correct += pred.eq(data.y[:, 0].long()).sum().item()
            if (perd_label > 0) and (data.y[:, 0].long() > 0):
                true_positives += 1
            elif (perd_label > 0) and (data.y[:, 0].long() == 0):
                false_positives += 1
            elif (perd_label == 0) and (data.y[:, 0].long() > 0):
                false_negatives += 1

        acc = correct / len(loader.dataset)
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = (2 * precision * recall) / (precision + recall)

        if val:
            print(f'Epoch {epoch} validation accuracy: {acc}')
            print(f'Epoch {epoch} validation precision: {precision}')
            print(f'Epoch {epoch} validation recall: {recall}')
            print(f'Epoch {epoch} validation f1 score: {f1_score}')
        else:
            print(f'Test accuracy: {acc}')
            print(f'Test precision: {precision}')
            print(f'Test recall: {recall}')
            print(f'Test f1 score: {f1_score}')

    return acc, precision, recall, f1_score


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
    target_class = 'gender'
    task = 'classification'
    numb_epochs = 1
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

    best_val_acc = 0.0

    # MAIN TRAINING LOOP
    for epoch in range(1, numb_epochs + 1):
        start = time.time()
        train(model, train_loader, epoch, device,
              optimizer, scheduler, writer)

        val_acc = test_classification(model, val_loader,
                                      indices['Val'], device,
                                      recording, results_folder,
                                      epoch=epoch)

        if recording:
            writer.add_scalar('Acc/val', val_acc, epoch)

            end = time.time()
            print('Time: ' + str(end - start))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), model_dir + '/model_best.pt')
                print('Saving Model'.center(60, '-'))
            writer.add_scalar('Time/epoch', end - start, epoch)

    test_classification(model, test_loader, indices['Test'], device, recording, results_folder, val=False)

    if recording:
        # save the last model
        torch.save(model.state_dict(), model_dir + '/model_last.pt')

        # Eval best model on test
        model.load_state_dict(torch.load(model_dir + '/model_best.pt'))

        with open(results_folder + '/results.csv', 'a', newline='') as results_file:
            result_writer = csv.writer(results_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            result_writer.writerow(['Best model!'])

        test_classification(model, test_loader, indices['Test'], device, recording, results_folder, val=False)
