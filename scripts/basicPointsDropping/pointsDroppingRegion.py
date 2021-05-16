import os.path as osp
PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..')
import sys
sys.path.append(PATH_TO_ROOT)
import numpy as np
import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet.src.utils import get_comment, get_data_path, data
from batchObject import BatchObject

from models.pointnet.src.models.pointnet2_regression_v2 import Net
#from models.pointnet.src.models.pointnet2_classification import Net

PATH_TO_POINTNET = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'models', 'pointnet') + '/'

def drop_points_region():
    print("num_labels")
    print(num_labels)
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.eval()

    for data in test_loader:
        data = data.to(device)
        original_pred = model(data)
        # print("original_pred")
        # print(original_pred)
        # print("original_pred shape")
        # print(original_pred.shape)

        # print("data")
        # print(data)
        # print("data.batch")
        # print(data.batch)
        # print("data.x")
        # print(data.x)
        num_points = data.pos.size(0)
        seg_regions = data.x[:, 3].long()
        # print("seg_regions shape")
        # print(seg_regions.shape)
        # print("seg_regions")
        # print(seg_regions)
        max_importance = 0
        most_important_label = -1
        for label in range(num_labels):
            drop_indices_array = []
            for n in range(num_points):
                # print("seg_regions[n]")
                # print(seg_regions[n])
                # print("seg_regions[n] shape")
                # print(seg_regions[n].shape)
                if seg_regions[n].item() - 1 == label:
                    drop_indices_array.append(n)
            residual_x = torch.from_numpy(np.delete(data.x.detach().cpu().numpy(), drop_indices_array, axis=0))
            residual_pos = torch.from_numpy(np.delete(data.pos.detach().cpu().numpy(), drop_indices_array, axis=0))
            residual_batch = torch.from_numpy(np.delete(data.batch.detach().cpu().numpy(), drop_indices_array, axis=0))
            residual_data = BatchObject(residual_x.to(device), residual_pos.to(device), residual_batch.to(device))
            residual_pred = model(residual_data)
            # print("residual_pred")
            # print(residual_pred)
            # print("residual_pred shape")
            # print(residual_pred.shape)
            importance = abs(residual_pred.item() - original_pred.item())
            if importance > max_importance:
                max_importance = importance
                most_important_label = label

        print("max_importance")
        print(max_importance)
        print("most important label")
        print(most_important_label)
        break


#Currently doing points dropping for Pointnet++ regression model
if __name__ == "__main__":

    local_features = ['corrected_thickness', 'curvature', 'sulcal_depth', 'segmentation']
    global_features = []

    recording = True
    REPROCESS = True

    data_nativeness = 'native'
    data_compression = "10k"
    data_type = 'white'
    hemisphere = 'both'

    # data_nativeness = 'native'
    # data_compression = "20k"
    # data_type = 'white'
    # hemisphere = 'left'

    additional_comment = ''

    #experiment_name = f'{data_nativeness}_{data_type}_{data_compression}_{hemisphere}_{additional_comment}'

    #experiment_name = 'native_white_10k_both_'
    #################################################
    ############ EXPERIMENT DESCRIPTION #############
    #################################################

    # 1. Model Parameters
    ################################################
    # lr = 0.001
    # batch_size = 2
    # gamma = 0.9875
    # scheduler_step_size = 2
    # target_class = 'scan_age'
    # task = 'regression'
    # numb_epochs = 200
    # number_of_points = 10000

    lr = 0.001
    batch_size = 1
    gamma = 0.9875
    scheduler_step_size = 2
    target_class = 'gender'
    task = 'classification'
    numb_epochs = 200
    number_of_points = 10000
    ################################################



    ###### SPECIFY PATH TO YOUR DATA_SPLIT PICKLE #####
    # 2. Get the data splits indices
    with open(PATH_TO_POINTNET + 'src/names.pk', 'rb') as f:
        indices = pickle.load(f)



    # 4. Get experiment description
    comment = get_comment(data_nativeness, data_compression, data_type, hemisphere,
                          lr, batch_size, local_features, global_features, target_class)

    print('=' * 50 + '\n' + '=' * 50)
    print(comment)
    print('=' * 50 + '\n' + '=' * 50)

    ##### SPECIFY YOUR DATA_FOLDER AND FILES_ENDING #####
    # 5. Perform data processing.
    data_folder, files_ending = get_data_path(data_nativeness, data_compression, data_type, hemisphere=hemisphere)

    train_dataset, test_dataset, validation_dataset, train_loader, test_loader, val_loader, num_labels = data(  data_folder,
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

    # 6. Getting the number of features to adapt the architecture
    try:
        num_local_features = train_dataset[0].x.size(1)
    except:
        num_local_features = 0
    print('Unique labels found: {}'.format(num_labels))

    num_global_features = len(global_features)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(num_local_features, num_global_features=num_global_features).to(device)

    #PATH = PATH_TO_POINTNET + 'experiment_data/new/{}-99/best_acc_model.pt'.format(experiment_name)

    # PATH = PATH_TO_ROOT + '/pointnetModels/classification/model_best.pt'

    #PATH = PATH_TO_ROOT + '/runs/classification/pointcloud_grad_cam/models/model_best.pt'
    PATH = PATH_TO_ROOT + '/runs/regression/Pointcloud_Grad_Cam/models/model_best.pt'

    drop_points_region()