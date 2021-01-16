import os.path as osp

PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', '..')
import sys

sys.path.append(PATH_TO_ROOT)

import os
import time
import pickle
import csv
import matplotlib.pyplot as plt

from models.pointnet.src.utils import get_data_path, data

PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', '..') + '/'
PATH_TO_POINTNET = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', '..', 'models', 'pointnet') + '/'

if __name__ == '__main__':

    ########## INDICES FOR DATA SPLIT #############
    with open(PATH_TO_POINTNET + 'src/names.pk', 'rb') as f:
        indices = pickle.load(f)
    ###############################################

