import numpy as np
import pandas as pd
# import pandas.DataFrame as dataFrame
import matplotlib.pyplot as plt
import os.path as osp
import csv
import time
import pickle

PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..')
import sys

sys.path.append(PATH_TO_ROOT)

PATH_TO_FILE = osp.dirname(osp.realpath(__file__))
path = PATH_TO_FILE + '/meta_data.tsv'
PATH_TO_POINTNET = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'models', 'pointnet') + '/'

# def read_meta(path=path):
#     '''Correctly reads a .tsv file into a numpy array'''
#     data = []
#
#     with open(path) as fd:
#         rd = csv.reader(fd, delimiter="\t", quotechar='"')
#         data = []
#         for idx, row in enumerate(rd):
#             if idx == 0:
#                 continue
#             data.append(row)
#
#     data = np.array(data)
#
#     return data

def split_patient_idx(patient_idx):
    patient_id, session_id = patient_idx.split('_')
    return patient_id, session_id

def read_meta(path=path):
    '''Correctly reads a .tsv file into a numpy array'''
    df = pd.read_csv(path, sep='\t', header=0)
    return df

def get_column(idxs=None, attribute=None):
#def plot_histogram(attribute):
    # print("indices")
    # print(indices)
    #meta_column_idx = categories[attribute]
   # attribute_vals = meta[:, meta_column_idx]
    #attribute_vals = meta[indices, meta_column_idx]
    #plt.hist(attribute_vals, bins='auto', label=trainTestVal)
    if idxs is None:
        #column = meta_df[attribute]
        idxs = indices['Train'] + indices['Val'] + indices['Test']
    meta_df_subset = meta_df[meta_df.patient_idx.isin(idxs)]
    column = meta_df_subset[attribute]

    return column
    # plt.grid()
    # plt.hist(attribute_vals, bins=10)
    #
    # plt.title("Histogram for {}".format(attribute))
    # image_fp = hist_path_root + attribute + ".png"
    # plt.savefig(image_fp)

if __name__ == '__main__':

    hist_path_root = PATH_TO_ROOT + "/histograms/"
    meta_df = read_meta()
    # Metadata categories
    categories = {'gender': 2, 'birth_age': 3, 'weight': 4, 'scan_age': 6, 'scan_num': 7}
    list_attrs = ['gender', 'scan_age', 'birth_age']

    attribute = 'birth_age'

    with open(PATH_TO_POINTNET + 'src/names.pk', 'rb') as f:
        indices = pickle.load(f)

    # for i in range(len(list_attrs)):
    #     attr = list_attrs[i]
    #     plot_histogram(attr)
    #     # plot_histogram("Train", indices['Train'], attr)
    #     # plot_histogram("Test", indices['Test'], attr)
    #     # plot_histogram("Val", indices['Val'], attr)
    #
    #     # plt.title("Histogram for {}".format(attr))
    #     # image_fp = hist_path_root + attr + ".png"
    #     # plt.savefig(image_fp)
    # bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #
    # meta_column_idx = categories[attribute]
    # attribute_vals = meta[:, meta_column_idx]
    # #attribute_vals = meta[indices, meta_column_idx]
    # #plt.hist(attribute_vals, bins='auto', label=trainTestVal)
    # plt.grid()
    # plt.hist(attribute_vals, bins=bins)
    #
    # plt.title("Histogram for {}".format(attribute))
    # indices_train = indices['Train']
    #print(indices_train)
    # pat_sess_ids_idxs = list(map(split_patient_idx, indices_train))
    # print(pat_sess_ids_idxs)

    meta_df["patient_idx"] = meta_df["participant_id"] + "_" + meta_df["session_id"].astype(str)
    # print("meta_df")
    # print(meta_df)

    plt.figure(figsize=(8, 6))
    all_column = get_column(attribute=attribute)
    plt.hist(all_column, bins=25, alpha=0.5, label="All")
    train_column = get_column(idxs=indices['Train'], attribute=attribute)
    plt.hist(train_column, bins=25, alpha=0.5, label="Train")
    # train_ax = train_column.plot.hist(bins=25)
    val_column = get_column(idxs=indices['Val'], attribute=attribute)
    plt.hist(val_column, bins=25, alpha=0.5, label="Val")
    #val_ax = val_column.plot.hist(bins=25)
    test_column = get_column(idxs=indices['Test'], attribute=attribute)
    plt.hist(test_column, bins=25, alpha=0.5, label="Test")
    # all_column = get_column(attribute=attribute)
    # plt.hist(all_column, bins=2, alpha=0.5, label="All")
    # print("all_column min")
    # print(all_column.min())
    # print("all_column max")
    # print(all_column.max())
    print("all attribute column num. elems")
    print(all_column.size)
    print("train attribute column num. elems")
    print(train_column.size)
    print("val attribute column num. elems")
    print(val_column.size)
    print("test attribute column num. elems")
    print(test_column.size)
    print("Sum of sizes")
    sum = train_column.size + val_column.size + test_column.size
    print(sum)
    #
    print("indices")
    print(indices)
    print("len(indices)")
    print(len(indices))
    print("number of training surfaces")
    print(len(indices['Train']))
    print("number of validation surfaces")
    print(len(indices['Val']))
    print("number of test surfaces")
    print(len(indices['Test']))
    sum_indices = len(indices['Train']) + len(indices['Val']) + len(indices['Test'])
    print("total number of surfaces")
    print(sum_indices)

    #test_ax = test_column.plot.hist(bins=25)

    # meta_df_train = meta_df[meta_df.patient_idx.isin(indices_train)]
    # print("meta_df_train")
    # print(meta_df_train)

    #x_vals_scan_age = np.arange(22.5, 47.5, 2.5)
    x_vals_birth_age = np.arange(22.5, 47.5, 2.5)
    # y_vals_gender = np.arange(0, 500, 50).tolist()
    image_fp = hist_path_root + attribute + "_TrainTestValAll.png"
    # plt.savefig(image_fp)
    # column = meta_df_train[attribute]
    # print("column")
    # print(column)
    # ax = column.plot.hist(bins=25)
    # train_ax.set_xlabel("Scan age (weeks)")
    # train_ax.set_xticks(x_vals_scan_age)

    plt.grid(True)
    #plt.xlabel("Gender")
    #plt.xlabel("Scan age (weeks)")
    plt.xlabel("Birth age (weeks)")
    plt.ylabel("Frequency")
    plt.xticks(x_vals_birth_age)
    plt.legend(loc='upper left')
    plt.savefig(image_fp)
    # column = column.astype(float)
    #ax = column.value_counts().plot(kind='bar', rot=0)
    #ax = column.plot.hist(x=attribute)
    # ax.set_xlabel("Gender")
    # ax.set_ylabel("Frequency")
    # ax.set_yticks(y_vals_gender)

    # train_ax.grid('on', axis='x')
    # train_ax.grid('on', axis='y')
    # fig = train_ax.get_figure()
    # fig.savefig(image_fp)