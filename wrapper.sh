#!/bin/bash -l

#$ -N pixelnerf-plants
#$ -l h_vmem=500G
#$ -l g=1

python_enable
source activate pixelnerf
cd pixel-nerf
python train/train.py -n co3d -c conf/exp/co3d.conf -B 2 -V 3 -D ./temp/CO3D --gpu_id=0