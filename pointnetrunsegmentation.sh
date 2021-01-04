#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr4617
#SBATCH --output="pointnetsegmentation-%j.out"
export PATH=/vol/bitbucket/sr4617/ForkedBrainSurfaceTK/venv/bin:$PATH
source activate
source /vol/cuda/10.1.105-cudnn7.6.5.32/setup.sh
TERM=vt100
/usr/bin/nvidia-smi
uptime

#pip install -r requirements1.txt -f https://download.pytorch.org/whl/torch_stable.html
#pip install -r requirements2.txt -f https://pytorch-geometric.com/whl/torch-1.5.0.html
python3 ./scripts/segmentation/PointNet/run_pointnet_segmentation.py
