import os
import os.path as osp

PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..')
import sys

sys.path.append(PATH_TO_ROOT)

import numpy as np
import vtk
import pyvista as pv
from vtk.util import numpy_support
from vtk.numpy_interface import dataset_adapter as dsa

def add_node_saliency_scores_to_vtk(saliency_scores, vtk_root, subject):
    saliency_scores_numpy = saliency_scores.detach().cpu().numpy()
    saliency_scores_numpy = np.squeeze(saliency_scores_numpy)
    original_vtk_file_name = vtk_root + "/" + subject + ".vtk"
    original_mesh = pv.read(original_vtk_file_name)
    # print("original_mesh.n_points")
    # print(original_mesh.n_points)
    # print("original_mesh.point_arrays")
    # print(original_mesh.point_arrays)
    original_mesh.point_arrays['saliency score'] = saliency_scores_numpy
    # print("original_mesh.point_arrays")
    # print(original_mesh.point_arrays)
    appended_vtk_file_name = "/vol/bitbucket/sr4617/ForkedBrainSurfaceTK/gcnRegressionSaliencyScores/" \
                             + subject + "_population_saliency_scores.vtk"
    original_mesh.save(appended_vtk_file_name)

# def add_node_saliency_scores_to_vtk_populn(saliency_scores, vtk_root):
#     saliency_scores_numpy = saliency_scores.detach().cpu().numpy()
#     saliency_scores_numpy = np.squeeze(saliency_scores_numpy)
#     original_vtk_file_name = vtk_root + "/" + subject + ".vtk"
#     original_mesh = pv.read(original_vtk_file_name)
#     original_mesh.point_arrays['saliency score'] = saliency_scores_numpy
#     appended_vtk_file_name = "/vol/bitbucket/sr4617/ForkedBrainSurfaceTK/gcnRegressionSaliencyScores/" \
#                              + subject + "_saliency_scores.vtk"
#     original_mesh.save(appended_vtk_file_name)
