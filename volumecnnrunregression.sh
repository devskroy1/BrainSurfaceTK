#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr4617
export PATH=/vol/bitbucket/sr4617/ForkedBrainSurfaceTK/venv/bin:$PATH
source activate
source /vol/cuda/10.1.105-cudnn7.6.5.32/setup.sh
TERM=vt100
/usr/bin/nvidia-smi
uptime

python3 -m scripts.regression.VolumeCNN.run_volume3d_regression
