import os
import os.path as osp

PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..')
import sys

sys.path.append(PATH_TO_ROOT)

import numpy as np
import vtk
from vtk.util import numpy_support
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
    #Type of mesh: vtkPolyData
    #point_data = mesh.GetPointData()
    #Type of point_data: vtkPointData

    mesh_new = dsa.WrapDataObject(mesh)

    mesh_new.PointData.SetAttribute(saliency_scores_numpy, 0)
    #mesh.PointData.append(saliency_scores_numpy, "saliency score")
    writer = vtk.vtkPolyDataWriter()
    appended_vtk_file_name = "/vol/bitbucket/sr4617/ForkedBrainSurfaceTK/gcnRegressionSaliencyScores/" + subject + "_saliency_scores.vtk"
    writer.SetFileName(appended_vtk_file_name)
    writer.SetInputData(mesh_new.VTKObject)
    writer.Write()

    # mesh_new = dsa.WrapDataObject(mesh)
    #
    # mesh_new.GetPointData().append(saliency_scores_numpy, "saliency score")
    # writer = vtk.vtkPolyDataWriter()
    # appended_vtk_file_name = "/vol/bitbucket/sr4617/ForkedBrainSurfaceTK/gcnRegressionSaliencyScores/" + subject + "_saliency_scores.vtk"
    # writer.SetFileName(appended_vtk_file_name)
    # writer.SetInputData(mesh_new.VTKObject)
    # writer.Write()

    # vtk_saliency_scores = numpy_support.numpy_to_vtk(saliency_scores_numpy, 1)
    # vtk_saliency_scores.SetName('point saliency score')
    # point_data_extra = point_data.AddArray(vtk_saliency_scores)
    # point_data_new = dsa.WrapDataObject(point_data_extra)
    # #Type of point_data_new: vtkPolyData
    # writer = vtk.vtkPolyDataWriter()
    # appended_vtk_file_name = "/vol/bitbucket/sr4617/ForkedBrainSurfaceTK/gcnRegressionSaliencyScores/" + subject + "_saliency_scores.vtk"
    # writer.SetFileName(appended_vtk_file_name)
    #
    # #writer.SetInputData(point_data.VTKObject)
    # writer.SetInputData(point_data_new.VTKObject)
    #
    # writer.Write()






    # mesh_new = dsa.WrapDataObject(mesh)
    # mesh_new.PointData.append(saliency_scores_numpy, "Saliency Score")
    # writer = vtk.vtkPolyDataWriter()
    # appended_vtk_file_name = "/vol/bitbucket/sr4617/ForkedBrainSurfaceTK/gcnRegressionSaliencyScores/" + subject + "_saliency_scores.vtk"
    # writer.SetFileName(appended_vtk_file_name)
    # writer.SetInputData(mesh_new.VTKObject)
    # writer.Write()