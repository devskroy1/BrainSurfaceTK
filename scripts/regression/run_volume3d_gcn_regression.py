import os
import os.path as osp
PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', '..')
import sys
sys.path.append(PATH_TO_ROOT)
import torch
from models.volume3d.utils.utils import read_meta, clean_data, split_data, get_ids_and_ages, plot_preds
import os.path as osp
from models.volume3d.main.train_validate import train_validate, save_to_log
from models.volume3d.main.train_test import train_test, save_to_log_test
from torch.utils.tensorboard import SummaryWriter
from models.volume3d.main.main import create_subject_folder
import math
import dgl
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.gNNs.data_utils import BrainNetworkDataset

cuda_dev = '0'  # GPU device 0 (can be changed if multiple GPUs are available)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:" + cuda_dev if use_cuda else "cpu")
print('Device: ' + str(device))
if use_cuda:
    print('GPU: ' + str(torch.cuda.get_device_name(int(cuda_dev))))

PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', '..') + '/'
PATH_TO_GNN = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'models', 'gNNs') + '/'
PATH_TO_VOLUME3D = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', '..', 'models', 'volume3d') + '/'

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    subjects, graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return np.array(subjects).reshape(len(graphs), -1), batched_graph, torch.tensor(labels).view(len(graphs), -1)

if __name__ == '__main__':

    load_path = os.path.join(
        "/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_native_04152020/merged/reducedto_10k/white/vtk")
    meta_data_file_path = os.path.join("/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/combined.tsv")

    save_path = "/vol/bitbucket/sr4617/tmp_save/vol_cnn_gcn_regression"

    lr = 0.001
    T_max = 10
    eta_min = 1e-6
    local_features = ('corrected_thickness', 'curvature', 'sulcal_depth')

    #writer = SummaryWriter(comment="segmentationbasicgcn")
    batch_size = 2
    train_test_split = (0.8, 0.1, 0.1)

    print("Batch size: ")
    print(batch_size)
    gcn_split_pk_fp = PATH_TO_GNN + 'names.pk'
    train_dataset_gcn = BrainNetworkDataset(load_path, meta_data_file_path, save_path=save_path, max_workers=8,
                                        dataset="train", train_split_per=train_test_split,
                                        index_split_pickle_fp=gcn_split_pk_fp, features=local_features)
    val_dataset_gcn = BrainNetworkDataset(load_path, meta_data_file_path, save_path=save_path, max_workers=8,
                                       dataset="val", features=local_features)
    test_dataset_gcn = BrainNetworkDataset(load_path, meta_data_file_path, save_path=save_path, max_workers=8,
                                       dataset="test", features=local_features)

    print("Building dataloaders")
    train_dl_gcn = DataLoader(train_dataset_gcn, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=8)
    val_dl_gcn = DataLoader(val_dataset_gcn, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=8)
    test_dl_gcn = DataLoader(test_dataset_gcn, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=8)

    additional_comment = ''

    # 1. What are you predicting?
    categories = {'gender': 3, 'birth_age': 4, 'weight': 5, 'scan_age': 7, 'scan_num': 8}
    meta_column_idx = categories['scan_age']

    # 2. Read the data and clean it
    meta_data = read_meta()

    ## 3. Get a list of ids and ages (labels)
    # ids, ages = get_ids_and_ages(meta_data, meta_column_idx)

    # 4. Set the parameters for the data pre-processing and split
    ################################
    ################################

    spacing = [3, 3, 3]
    image_size = [60, 60, 50]
    smoothen = 8
    edgen = False
    test_size = 0.09
    val_size = 0.1
    random_state = 42
    REPROCESS = False

    ################################
    ################################


    # 4. Create subject folder
    fn, counter = create_subject_folder()
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', f'{fn}/')

    # 5. Split the data
    dataset_train, dataset_val, dataset_test = split_data(meta_data,
                                                          meta_column_idx,
                                                          spacing,
                                                          image_size,
                                                          smoothen,
                                                          edgen,
                                                          val_size,
                                                          test_size,
                                                          path=path,
                                                          reprocess=REPROCESS)

    # 6. Create CNN Model parameters
    ################################
    ################################
    ################################

    USE_GPU = True
    dtype = torch.float32
    num_of_parameters_multiplier = 10
    num_epochs = 200
    #lr = 0.006882801723742766
    gamma = 0.97958263796472
    #batch_size = 32
    dropout_p = 0.5
    scheduler_frequency = 3

    ################################
    ################################
    ################################

    # 6. Create tensorboard writer
    writer = SummaryWriter(PATH_TO_VOLUME3D + f'tensorboard_runs/Subject{additional_comment}-{counter}')

    # 7. Run TRAINING + VALIDATION after every N epochs
    model, params, final_MAE = train_validate(lr, num_of_parameters_multiplier, num_epochs,
                                              gamma, batch_size,
                                              dropout_p, dataset_train,
                                              dataset_val, fn, counter,
                                              scheduler_frequency,
                                              writer, train_dl_gcn, val_dl_gcn)

    # 8. Save the results
    save_to_log(model, params,
                fn, final_MAE,
                num_epochs,
                batch_size,
                lr, num_of_parameters_multiplier,
                gamma, smoothen,
                edgen, dropout_p,
                spacing, image_size,
                scheduler_frequency)


    """# Full Train & Final Test"""

    # 2. Create TEST folder
    fn, counter = create_subject_folder(test=True)

    # 3. Run TRAINING + TESTING after every N epochs
    model, params, score, train_loader, test_loader = train_test(lr, num_of_parameters_multiplier, num_epochs, gamma,
                                                                 batch_size, dropout_p,
                                                                 dataset_train, dataset_test,
                                                                 fn, counter,
                                                                 scheduler_frequency,
                                                                 writer, train_dl_gcn, test_dl_gcn)

    # 4. Record the TEST results
    save_to_log_test(model, params, fn, score, num_epochs, batch_size,
                     lr, num_of_parameters_multiplier, gamma, smoothen, edgen, dropout_p, spacing,
                     image_size, scheduler_frequency)


    # 5. Perform the final testing
    model.eval()
    pred_ages = []
    actual_ages = []
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device=device)  # move to device, e.g. GPU
            batch_labels = batch_labels.to(device=device)
            batch_preds = model(batch_data)

            pred_ages.append([batch_preds[i].item() for i in range(len(batch_preds))])
            actual_ages.append([batch_labels[i].item() for i in range(len(batch_labels))])

    plot_preds(pred_ages, actual_ages, writer, num_epochs, test=True)
