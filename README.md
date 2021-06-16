# BrainSurfaceToolKit

This repository is forked from the BrainSurfaceTK Github repository at https://github.com/andwang1/BrainSurfaceTK, developed by Andy Wang, Alex Zakharov, 
Cemlyn Waters and Amir Alansary. The basic PointNet++, GCN, MeshCNN and VolumeCNN models' code is from their repository.
<div align="center"> 

<img src="https://github.com/andwang1/BrainSurfaceTK/blob/master/GUI/main/static/main/gifs/rotate-big.gif?raw=true" width="600" height="450"/>
</div>

# Setting up
To install all required packages, please setup a virtual environment as per the instructions below. This virtual environment is based on a CUDA 10.1.105 installation.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements1.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements2.txt -f https://pytorch-geometric.com/whl/torch-1.5.0.html
```

Alternatively, for a CPU installation, please setup the virtual environment as per the instructions below. Please note that the MeshCNN model requires the CUDA based installation above.
```
python3 -m venv venv
source venv/bin/activate
pip install -r cpu_requirements1.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install -r cpu_requirements2.txt -f https://pytorch-geometric.com/whl/torch-1.5.0.html
``` 

# PointNet++
PointNet++ is a hierarchical neural network, which was proposed to be used on point-cloud geometric data [1] for the tasks of regression, classification, and segmentation. In this project, we apply this architecture onto point-cloud representations of brain surfaces to tackle the tasks of age regression and brain segmentation.

###### Run instructions

The run instructions differ slightly for Pointnet regression and segmentation. Please proceed to the README in models/pointnet of this repository for full information.


# GCNN

GCNN [3] is a Graph Convolution Neural Network and uses the Deep Graph Library (DGL) [4] implementation of a Graph Convolutional layer.

###### Run instructions

Here is an example on how to run the model:
```
python -u models/gNNs/basicgcntrain.py /path_to/meshes False all --batch_size 32 --save_path ../tmp_save --results ./results
```
Please note that the BrainNetworkDataset will convert the vtk PolyData and save them as DGL graphs in a user-specified
folder. This is don't because the conversion process can be a bit slow and for multiple experiments, this becomes beneficial.

# Happy Researching!

<div align="center"> 

<img src="https://github.com/andwang1/BrainSurfaceTK/blob/master/img/CC00380XX10_121200.gif?raw=true" width="600" height="450"/>
</div>


# Acknowledgements
The code of RandLA-Net is based on the RandLA-Net-pytorch Github repository at https://github.com/aRI0U/RandLA-Net-pytorch, developed by Alain Riou and Thibaud-Ardoin.
The code of PointASNL is based on the PointASNL Github repository at https://github.com/yanx27/PointASNL, developed by Xu Yan.
The code of PointNet CAM is based on the Pointcloud-grad-CAM Github repository at https://github.com/Fragjacker/Pointcloud-grad-CAM, developed by Dennis Struhs.
See the LICENSE file in the root directory on the master branch for their licenses.
