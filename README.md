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
PointNet++ is a hierarchical neural network, which was proposed to be used on point-cloud geometric data [1] for the tasks of regression, classification, and segmentation. In this project, we apply this architecture onto point-cloud representations of brain surfaces to tackle the tasks of age regression, brain segmentation, sex classification and preterm classification. We add Local Feature Aggregation (LFA) units from RandLA-Net for segmentation and classification tasks and Adaptive Sampling (AS) and Point Local-Point Non-Local (PL-PNL) cells from PointASNL for classification and regression tasks. We also try to improve PointNet++ classification results by applying dGCNN, using extra local and global features, varying the architecture (by varying the number of Set Abstraction layers for instance), tuning hyperparameters such as the Learning Rate and including a separate pathway consisting of 3D Convolutional layers from the Volume CNN. 

###### Run instructions

The run instructions differ slightly for Pointnet regression and segmentation. Please proceed to the README in models/pointnet of this repository for full information.

# GCNN

GCNN [3] is a Graph Convolution Neural Network and uses the Deep Graph Library (DGL) [4] implementation of a Graph Convolutional layer. There are segmentation and regression architectures. We explore variations of the GCNN by introducing Dynamic Edge Convolution from dGCNN into the segmentation network. For the regression network, we also carry out hyperparameter tuning and vary the architecture (varying the number of Graph Convolution layers for instance). We also introduce different pooling layers such as Sort and Max pooling into the regression network.

###### Run instructions

Here is an example on how to run the model:
```
python -u models/gNNs/basicgcntrain.py /path_to/meshes False all --batch_size 32 --save_path ../tmp_save --results ./results
```
Please note that the BrainNetworkDataset will convert the vtk PolyData and save them as DGL graphs in a user-specified
folder. This is don't because the conversion process can be a bit slow and for multiple experiments, this becomes beneficial.

# MeshCNN

MeshCNN [2] is a general-purpose deep neural network for 3D triangular meshes, which can be used for tasks such as 3D shape classification or segmentation. 
This framework includes convolution, pooling and unpooling layers which are applied directly on the mesh edges.

The original GitHub repo and additional run instructions can be found here: https://github.com/ranahanocka/MeshCNN/

In this repository, we have made multiple modifcations. These include functionality for regression, adding global features into the penultimate fully-connected layers, adding logging of test-ouput, allowing for a train/test/validation split, and functionality for new learning-rate schedulers among other features.

###### Run instructions

Place the .obj mesh data files into a folder in *models/MeshCNN/datasets* with the correct folder structure - below is an example of the structure. Here, *brains* denotes the name of the directory in *models/MeshCNN/datasets* which holds one directory for each class, here e.g. *Male* and *Female*.
In each class, folders *train*, *val* and *test* hold the files.

<img src="https://github.com/andwang1/BrainSurfaceTK/blob/master/img/meshcnn_data.png?raw=true" width="450" height="263" />

Please additionally place a file called *meta_data.tsv* in the *models/MeshCNN/util* folder. This tab-seperated file will be used to read in additional labels and features into the model.
The file should contain columns participant_id and session_id, which will be concatenated to form a unique identifier of a patient's scan. This unique identifier must be used to name the data files in the datasets/ folder structure described above.
E.g. a *meta_data.tsv* file might look like this:

participant_id	session_id	scan_age

CC00549XX22	100100	42.142347

The corresponding mesh data file must then be named
*CC00549XX22_100100.obj*

Any continuous-valued columns in the *meta_data.tsv* file can then be used as features or labels in the regression using switches in the training file, as mentioned below.
```
--label scan_age
--features birth_age
```

From the main repository level, the model can then be trained using, e.g. for regression
```
./scripts/regression/MeshCNN/train_reg_brains.sh
```
Similarly, a pretrained model can be applied to the test set, e.g.
```
./scripts/regression/MeshCNN/test_reg_brains.sh
```

# Volume CNN

Volume CNN is a 3D CNN applied to volumetric MRI images, using voxel based grids, for the task of age regression. We enhance the Volume CNN regression network architecture by including a separate pathway consisting of Graph Convolution layers from the GCNN, which we apply to graph based representations of surfaces.

###### Run instructions

Please refer to the README in models/volume3d of this repository for the run instructions

# Happy Researching!

<div align="center"> 

<img src="https://github.com/andwang1/BrainSurfaceTK/blob/master/img/CC00380XX10_121200.gif?raw=true" width="600" height="450"/>
</div>


# Acknowledgements
The code of RandLA-Net is based on the RandLA-Net-pytorch Github repository at https://github.com/aRI0U/RandLA-Net-pytorch, developed by Alain Riou and Thibaud-Ardoin.
The code of PointASNL is based on the PointASNL Github repository at https://github.com/yanx27/PointASNL, developed by Xu Yan.
The code of PointNet CAM is based on the Pointcloud-grad-CAM Github repository at https://github.com/Fragjacker/Pointcloud-grad-CAM, developed by Dennis Struhs.
See the LICENSE file in the root directory on the master branch for their licenses.
