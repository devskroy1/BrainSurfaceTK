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

def read_meta(path=path):
    '''Correctly reads a .tsv file into a numpy array'''
    df = pd.read_csv(path, sep='\t', header=0)
    return df

#def plot_histogram(trainTestVal, indices, attribute):
def plot_histogram(attribute):
    # print("indices")
    # print(indices)
    meta_column_idx = categories[attribute]
    attribute_vals = meta[:, meta_column_idx]
    #attribute_vals = meta[indices, meta_column_idx]
    #plt.hist(attribute_vals, bins='auto', label=trainTestVal)
    plt.grid()
    plt.hist(attribute_vals, bins=10)

    plt.title("Histogram for {}".format(attribute))
    image_fp = hist_path_root + attribute + ".png"
    plt.savefig(image_fp)

if __name__ == '__main__':

    hist_path_root = PATH_TO_ROOT + "/histograms/"
    meta_df = read_meta()
    # Metadata categories
    categories = {'gender': 2, 'birth_age': 3, 'weight': 4, 'scan_age': 6, 'scan_num': 7}
    list_attrs = ['gender', 'scan_age', 'birth_age']

    attribute = 'gender'

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
    x_vals_scan_age = np.arange(22.5, 47.5, 2.5).tolist()
    y_vals_gender = np.arange(0, 500, 50).tolist()
    image_fp = hist_path_root + attribute + ".png"
    # plt.savefig(image_fp)
    column = meta_df[attribute]
    # ax = column.plot.hist(bins=25)
    # ax.set_xlabel("Scan age")
    # ax.set_xticks(x_vals)

    # column = column.astype(float)
    ax = column.value_counts().plot(kind='bar', rot=0)
    #ax = column.plot.hist(x=attribute)
    ax.set_xlabel("Gender")
    ax.set_ylabel("Frequency")
    ax.set_yticks(y_vals_gender)
    # ax.grid('on', axis='x')
    ax.grid('on', axis='y')
    fig = ax.get_figure()
    fig.savefig(image_fp)