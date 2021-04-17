import os
import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import itertools
from ..src.data_loader import OurDataset
PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..') + '/'


def get_comment(data_nativeness, data_compression, data_type, hemisphere,
                lr, batch_size, local_feature_combo, global_features,
                target_class, log_descr=False):

    comment = '\nData nativeness: {}, \n'.format(data_nativeness) + \
              'Data compression: {}, \n'.format(data_compression) + \
              'Data type: {}, \n'.format(data_type) + \
              'Hemispheres: {}, \n'.format(hemisphere) + \
              'Local features: {}, \n'.format(local_feature_combo) + \
              'Global features: {}, \n'.format(global_features) + '\n' + \
              'Learning rate: {}, \n'.format(lr) + \
              'Batch size: {}\n'.format(batch_size)

    if log_descr == True:
        # 0. Save to log_record.txt
        comment = data_nativeness + '  ' + data_compression + "  " + data_type + '  ' + hemisphere + '  ' \
                    + "LR=" + str(lr) + '\t\t' \
                    + "Batch=" + str(batch_size) + '\t\t' \
                    + "Local features:" + str(local_feature_combo) + '\t\t' \
                    + "Global features:" + str(global_features) + '\t\t' \
                    + "Data used: " + data_compression + '_' + data_type + '\t\t' \
                    + "Split class: " + target_class

    return comment


def get_id(prefix=''):
    '''
    :return: The next expected id number of an experiment
             that hasn't yet been recorded!
    '''
    with open(PATH_TO_ROOT + 'logs/new/LOG_{}.txt'.format(prefix), 'r') as log_record:
        next_id = len(log_record.readlines()) + 1

    return str(next_id)


def save_to_log(experiment_description, prefix=''):
    '''
    Saves the experiment description into the log
    :param experiment_description: all the variables in a nice format
    '''

    # If the log file does not exist - create it
    add_first_line = False
    if not os.path.exists(PATH_TO_ROOT + 'logs/new/LOG_{}.txt'.format(prefix)):
        add_first_line = True
        filename = PATH_TO_ROOT + 'logs/new/LOG_{}.txt'.format(prefix)
        dirname = os.path.dirname(filename)
        os.makedirs(dirname)

    if add_first_line:
        with open(PATH_TO_ROOT + 'logs/new/LOG_{}.txt'.format(prefix), 'w+') as log_record:
            log_record.write('LOG OF ALL THE EXPERIMENTS for {}'.format(prefix))

    with open(PATH_TO_ROOT + 'logs/new/LOG_{}.txt'.format(prefix), 'a+') as log_record:
        next_id = get_id(prefix=prefix)
        log_record.write('\n#{} ::: {}'.format(next_id, experiment_description))


def get_grid_search_local_features(local_feats):
    '''
    Return all permutations of parameters passed
    :return: [ [], [cortical], [myelin], ..., [cortical, myelin], ... ]
    '''

    # 1. Get all permutations of local features
    local_fs = [[], ]
    for l in range(1, len(local_feats) + 1):
        for i in itertools.combinations(local_feats, l):
            local_fs.append(list(i))

    return local_fs


def get_data_path(data_nativeness, data_compression, data_type, hemisphere='left'):

    '''
    Returns the correct path to files

    data_nativeness: {'native', 'aligned'}
    data_compression: {'50', '90', '5k', '10k', '20k', '30k'}
    data_type: {'inflated', 'pial', 'midthickness', 'sphere', 'veryinflated', 'white'}
    hemisphere: {'left', 'right', 'both'}

    Eg.
    data_folder = '/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_fsavg32k/reduced_50/vtk/pial'
    data_folder = '/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_native/reduced_50/inflated/vtk'
    files_ending = '_hemi-L_pial_reduce50.vtk'
    files_ending = '_left_inflated_reduce50.vtk'

    For merged:
    file names : sub-CC00050XX01_ses-7201_merged_white.vtk
    path to files: /vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_native_04152020/merged/original_native/white/vtk

    # NATIVE
    left: /vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_native_04152020/hemispheres/reducedto_30k/inflated/vtk
    merged: /vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_native_04152020/merged/reducedto_30k/inflated/vtk
    surface_native_04152020/hemispheres/reducedto_30k/inflated/vtk/sub-CC00401XX05_ses-123900_merged_inflated.vtk


    left:   sub-CC00050XX01_ses-7201_left_inflated_30k.vtk
    merged: sub-CC00050XX01_ses-7201_merged_inflated_30k.vtk
    '''

    root = '/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/'


    data_nativeness_paths = {'native': 'surface_native_04152020/',
                             'aligned': 'surface_fsavg32k/'}

    hemispheres = {'both': 'merged/',
                  'left': 'hemispheres/',
                  'right': 'hemispheres/'}

    data_compression_paths = {'50': 'reduced_50/',
                              50: 'reduced_50/',
                              '90': 'reduced_90/',
                              90: 'reduced_90/',
                              'original_native': 'original_native/',
                              'original': 'original_32k/',
                              'original_aligned': 'original_32k/',
                              '5k': 'reducedto_05k/',
                              '10k': 'reducedto_10k/',
                              '20k': 'reducedto_20k/',
                              '30k': 'reducedto_30k/',
                              }


    data_type_paths = {'inflated': 'inflated/',
                       'pial': 'pial/',
                       'midthickness': 'midthickness/',
                       'sphere': 'sphere/',
                       'veryinflated': 'veryinflated/',
                       'white': 'white/'}

    hemisphere_paths = {'left_native_original': '_left_{}.vtk'.format(data_type),
                        'left_native_50':       '_left_{}_reducedby50percent.vtk'.format(data_type),
                        'left_native_90':       '_left_{}_reducedby90percent.vtk'.format(data_type),
                        'left_native_5k': '_left_{}_05k.vtk'.format(data_type),
                        'left_native_10k':      '_left_{}_10k.vtk'.format(data_type),
                        'left_native_20k':      '_left_{}_20k.vtk'.format(data_type),
                        'left_native_30k':      '_left_{}_30k.vtk'.format(data_type),

                        'right_native_original': '_right_{}.vtk'.format(data_type),
                        'right_native_50':       '_right_{}_reducedby50percent.vtk'.format(data_type),
                        'right_native_90':       '_right_{}_reducedby90percent.vtk'.format(data_type),
                        'right_native_5k': '_right_{}_05k.vtk'.format(data_type),
                        'right_native_10k':      '_right_{}_10k.vtk'.format(data_type),
                        'right_native_20k':      '_right_{}_20k.vtk'.format(data_type),
                        'right_native_30k':      '_right_{}_30k.vtk'.format(data_type),

                        'merged_native_original': '_merged_{}.vtk'.format(data_type),
                        'merged_native_50':       '_merged_{}_reducedby50percent.vtk'.format(data_type),
                        'merged_native_90':       '_merged_{}_reducedby90percent.vtk'.format(data_type),
                        'merged_native_5k': '_merged_{}_05k.vtk'.format(data_type),
                        'merged_native_10k':      '_merged_{}_10k.vtk'.format(data_type),
                        'merged_native_20k':      '_merged_{}_20k.vtk'.format(data_type),
                        'merged_native_30k':      '_merged_{}_30k.vtk'.format(data_type),

                        'left_aligned_original': '_hemi-L_{}.vtk'.format(data_type),
                        'left_aligned_50':       '_hemi-L_{}_reduce50.vtk'.format(data_type),
                        'left_aligned_90':       '_hemi-L_{}_reduce90.vtk'.format(data_type),

                        'right_aligned_original': '_hemi-R_{}.vtk'.format(data_type),
                        'right_aligned_50':       '_hemi-R_{}_reduce50.vtk'.format(data_type),
                        'right_aligned_90':       '_hemi-R_{}_reduce90.vtk'.format(data_type)}


    if hemisphere == 'both':
        _hemisphere = 'merged'
    else:
        _hemisphere = hemisphere


    if data_nativeness == 'native':

        data_folder = root + data_nativeness_paths[data_nativeness] + hemispheres[hemisphere] + data_compression_paths[data_compression] + data_type_paths[data_type] + 'vtk'
        files_ending = hemisphere_paths['{}_{}_{}'.format(_hemisphere, data_nativeness, data_compression)]

        return data_folder, files_ending


    elif data_nativeness == 'aligned':

        if data_compression == 'original':
            data_folder = root + data_nativeness_paths[data_nativeness] \
                          + data_compression_paths['{}_{}'.format(data_compression, data_nativeness)] \
                          + 'vtk/' + data_type_paths[data_type][:-1]
            files_ending = hemisphere_paths[hemisphere + '_{}_{}'.format(data_nativeness, data_compression)]

        else:
            data_folder = root + data_nativeness_paths[data_nativeness] \
                          + data_compression_paths[data_compression] \
                          + 'vtk/' + data_type_paths[data_type][:-1]
            files_ending = hemisphere_paths[hemisphere + '_{}_{}'.format(data_nativeness, data_compression)]

        return data_folder, files_ending



def data(data_folder, files_ending, data_type, target_class, task, REPROCESS, local_features, global_features, indices, batch_size, num_workers=2,
         data_compression=None, data_nativeness=None, hemisphere=None):
    '''
    Get data loaders and data sets

    :param data_folder:
    :param files_ending:
    :param data_type:
    :param target_class:
    :param task:
    :param REPROCESS:
    :param local_features:
    :param global_features:
    :param indices:
    :param batch_size:
    :param num_workers:
    :return:
    '''

    path = osp.join(
        osp.dirname(osp.realpath(__file__)), '..', 'data/' + 'segmentation' +
                    '/{}_{}_{}/{}'.format(data_compression, data_nativeness, hemisphere, data_type))

    # Transformations
    transform = T.Compose([
        # T.RandomTranslate(0.1),
        # T.RandomFlip(0, p=0.3),
        # T.RandomFlip(1, p=0.1),
        # T.RandomFlip(2, p=0.3),
        # T.FixedPoints(500, replace=False), #32492  16247
        T.RandomRotate(360, axis=0),
        T.RandomRotate(360, axis=1),
        T.RandomRotate(360, axis=2)
    ])

    pre_transform = T.NormalizeScale()
    print('Starting dataset processing...')
    train_dataset = OurDataset(path, train=True, transform=transform, pre_transform=pre_transform,
                               target_class=target_class, task=task, reprocess=REPROCESS,
                               local_features=local_features, global_feature=global_features,
                               val=False, indices=indices['Train'],
                               data_folder=data_folder,
                               files_ending=files_ending)

    test_dataset = OurDataset(path, train=False, transform=transform, pre_transform=pre_transform,
                              target_class=target_class, task=task, reprocess=REPROCESS,
                              local_features=local_features, global_feature=global_features,
                              val=False, indices=indices['Test'],
                              data_folder=data_folder,
                              files_ending=files_ending)

    validation_dataset = OurDataset(path, train=False, transform=transform, pre_transform=pre_transform,
                                    target_class=target_class, task=task, reprocess=REPROCESS,
                                    local_features=local_features, global_feature=global_features,
                                    val=True, indices=indices['Val'],
                                    data_folder=data_folder,
                                    files_ending=files_ending)

    num_labels = train_dataset.num_labels

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataset, test_dataset, validation_dataset, train_loader, test_loader, val_loader, num_labels



if __name__ == '__main__':
    pass
     # # Model Parameters
    # lr = 0.001
    # batch_size = 8
    # num_workers = 2
    #
    # local_features = ['corr_thickness', 'myelin_map', 'curvature', 'sulc']
    # global_features = None
    # target_class = 'gender'
    # task = 'segmentation'
    # # number_of_points = 12000
    #
    # test_size = 0.09
    # val_size = 0.1
    # reprocess = False
    #
    # data = "reduced_50"
    # type_data = "inflated"
    #
    # log_descr = "LR=" + str(lr) + '\t\t'\
    #           + "Batch=" + str(batch_size) + '\t\t'\
    #           + "Num Workers=" + str(num_workers) + '\t'\
    #           + "Local features:" + str(local_features) + '\t'\
    #           + "Global features:" + str(global_features) + '\t'\
    #           + "Data used: " + data + '_' + type_data + '\t'\
    #           + "Split class: " + target_class
    #
    # save_to_log(log_descr)