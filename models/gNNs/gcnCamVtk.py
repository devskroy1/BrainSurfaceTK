import os
import os.path as osp

PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..')
import sys

sys.path.append(PATH_TO_ROOT)

import numpy as np
import vtk
from vtk.numpy_interface import dataset_adapter as dsa

def add_node_saliency_scores_to_vtk(saliency_scores, vtk_root, subject):
    saliency_scores_numpy = saliency_scores.detach().cpu().numpy()
    original_vtk_file_name = vtk_root + "/" + subject + ".vtk"
    # print("Inside add_node_saliency_scores_to_vtk()")
    # print("original_vtk_file_name")
    # print(original_vtk_file_name)
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(original_vtk_file_name)
    reader.Update()
    mesh = reader.GetOutput()

    mesh_new = dsa.WrapDataObject(mesh)
    mesh_new.PointData.append(saliency_scores_numpy, "Saliency Score")
    writer = vtk.vtkPolyDataWriter()
    appended_vtk_file_name = "/vol/bitbucket/sr4617/ForkedBrainSurfaceTK/gcnRegressionSaliencyScores/" + subject + "_saliency_scores.vtk"
    writer.SetFileName(appended_vtk_file_name)
    writer.SetInputData(mesh_new.VTKObject)
    writer.Write()