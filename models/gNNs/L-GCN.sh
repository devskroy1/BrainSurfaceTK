#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cnw119
whoami
export PATH=/vol/bitbucket/${USER}/miniconda3/bin/:$PATH
source activate
conda activate vortexAI
# source /vol/cuda/10.0.130/setup.sh
TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
uptime
cd /vol/bitbucket/${USER}/neodeepbrain || exit


python -u -m models.gNNs.basicgcntrain left 10k featureless=True
python -u -m models.gNNs.basicgcntrain left 10k featureless=False
python -u -m models.gNNs.basicgcntrain left 20k featureless=True
python -u -m models.gNNs.basicgcntrain left 10k featureless=False
