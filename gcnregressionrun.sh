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

#python -u models/gNNs/basicgcntrain.py /vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_fsavg32k/reduced_50/vtk/white False some --batch_size 64 --save_path ../tmp_save --results ./results
python -u models/gNNs/basicgcntrain.py /vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_native_04152020/merged/reducedto_10k/pial/vtk False some --batch_size 2 --save_path ../tmp_save --results ./results
#python -u models/gNNs/basicgcntrain.py /vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_native_04152020/merged/reducedto_10k/white/vtk False some --batch_size 2 --save_path ../tmp_save --results ./results
#python -u models/gNNs/basicgcntrain.py /vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/surface_fsavg32k/reduced_90/vtk/white False some --batch_size 64 --save_path ../tmp_save --results ./results
#python -u models/gNNs/basicgcntrain.py /vol/bitbucket/sr4617/ForkedBrainSurfaceTK/alignedSurfacesWhite False some --batch_size 64 --save_path ../tmp_save --results ./results
#python -u models/gNNs/basicgcntrain.py /vol/bitbucket/sr4617/ForkedBrainSurfaceTK/alignedSurfacesPial False some --batch_size 64 --save_path ../tmp_save --results ./results
#python -u models/gNNs/basicgcntrain.py /vol/bitbucket/sr4617/ForkedBrainSurfaceTK/alignedSurfacesWhiteLeft False some --batch_size 64 --save_path ../tmp_save --results ./results
#python -u models/gNNs/basicgcntrain.py /vol/bitbucket/sr4617/ForkedBrainSurfaceTK/alignedSurfacesWhiteRight False some --batch_size 64 --save_path ../tmp_save --results ./results
#python -u models/gNNs/basicgcntrain.py /vol/bitbucket/sr4617/ForkedBrainSurfaceTK/alignedSurfacesPialLeft False some --batch_size 64 --save_path ../tmp_save --results ./results
#python -u models/gNNs/basicgcntrain.py /vol/bitbucket/sr4617/ForkedBrainSurfaceTK/alignedSurfacesPialRight False some --batch_size 64 --save_path ../tmp_save --results ./results
