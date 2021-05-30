import os
import numpy as np
import pyvista as pv
from read_meta import read_meta
import sys

from get_edge_features import write_eseg, write_seseg, save_features
from models.MeshCNN.models.layers.mesh_prepare import from_scratch

#pyvista docs: https://docs.pyvista.org/plotting/plotting.html#pyvista.BasePlotter.add_mesh

__author__ = "Francis Rhys Ward"
__license__ = "MIT"


if __name__ == '__main__':


    meta_data_path = sys.argv[1]
    vtk_path = sys.argv[2]
    path = sys.argv[3]
    try:
        seg = sys.argv[4] == "seg"
        num_labels = int(sys.argv[5])
    except:
        seg = False

    obj_path = path+"obj/"
    seg_path = path+"seg/"
    sseg_path = path+"sseg/"
    feat_path = path+"local_features/"

    extension = "_merged_white_10k.vtk"
    meta_data = read_meta(meta_data_path)
    patient_names = []
    ses_ids = []


    for idx, ids in enumerate(meta_data):
        patient_id, ses_id = ids[0], ids[1]
        print(patient_id, ses_id)

       # p = pv.Plotter(off_screen=False, notebook=False)
        p = pv.Plotter(off_screen=True)

        print("Got past plotter init")
        try:
            meshv = pv.read(vtk_path+"sub-"+patient_id+"_ses-"+ses_id+extension)
        except:
            print("failed")
            continue
        p.add_mesh(meshv, show_edges=True)
        print("Got past add_mesh")
        p.export_obj(obj_path+patient_id+"_"+ses_id)
        print("Got past export_obj")

        if seg:
            feature_arrays = {'drawem': 0, 'corr_thickness': 1, 'myelin_map': 2, 'curvature': 3, 'sulc': 4}
            vert_features = {feature:meshv.get_array(feature_arrays[feature]) for feature in feature_arrays.keys()}

            mesh_data = from_scratch(file=obj_path+ patient_id+"_"+ses_id+".obj", opt=None)

            write_eseg(mesh_data, vtk_path+"sub-"+patient_id+"_ses-"+  ses_id+extension, seg_path, patient_id, ses_id)

            try:
                write_seseg(seg_path, sseg_path, patient_id, ses_id, num_labels)
            except:
                continue
            save_features(mesh_data, vtk_path+"sub-"+patient_id+"_ses-"+ses_id+extension, feat_path, patient_id, ses_id)
