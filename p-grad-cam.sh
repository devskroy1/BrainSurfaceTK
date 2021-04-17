#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr4617
#SBATCH --output="pointnetclassification-%j.out"
export PATH=/vol/bitbucket/sr4617/ForkedBrainSurfaceTK/anaconda/anaconda3:$PATH
cd /vol/bitbucket/sr4617/ForkedBrainSurfaceTK/anaconda/anaconda3
source bin/activate
conda activate envs/p-grad-CAM
cd /vol/bitbucket/sr4617/ForkedBrainSurfaceTK/scripts/pointnetCAM

python3 pointnetCAM.py
