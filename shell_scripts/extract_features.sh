#!/bin/bash

#SBATCH --mail-user=pablooyarzo@zedat.fu-berlin.de
#SBATCH --job-name=extract_features
#SBATCH --mail-type=end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20000
#SBATCH --time=14:00:00
#SBATCH --qos=standard
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu


#down here goes the script
module add FFmpeg/4.1-foss-2018b
module add PyTorch/1.4.0-fosscuda-2019b-Python-3.7.4
module add torchvision/0.5.0-fosscuda-2019b-Python-3.7.4-PyTorch-1.4.0

python extract_features.py --inference --model resnet --model_depth 50 --n_classes 700 \
           --resume_path /scratch/kshitijd/Algonauts2020/checkpoints/r3d50_K_200ep.pth \
           --save_dir /scratch/kshitijd/Algonauts2020/activations/r3d50_K \
           --video_path /scratch/kshitijd/Algonauts2020/AlgonautsVideos268_All_30fpsmax/
