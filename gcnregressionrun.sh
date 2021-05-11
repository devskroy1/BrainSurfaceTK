#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr4617
#SBATCH --output="gcntrain-%j.out"
export PATH=/vol/bitbucket/sr4617/ForkedBrainSurfaceTK/venv/bin:$PATH
source activate
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
TERM=vt100
/usr/bin/nvidia-smi
uptime

python -u models/gNNs/basicgcntrain.py /vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_native_04152020/merged/reducedto_10k/white/vtk False some --batch_size 64 --lr 0.005 --save_path ../tmp_save --results ./results
