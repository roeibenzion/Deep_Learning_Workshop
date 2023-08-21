#!/bin/bash
#SBATCH --job-name=non_basic_aug
#SBATCH --partition=studentkillable
#SBATCH --gpus=4
#SBATCH --time=1440
#SBATCH --mem=64G
#SBATCH --output=./logs/non_basic_aug.out # redirect stdout
#SBATCH --error=./logs/non_basic_aug.err  # redirect stderr

module load cuda/12.0
source activate /home/yandex/DLW2023b/dimakisilev/myenv
python src/main.py --batch_size 4  --num_epochs 100 --steps_per_epoch 550 --root_path './data' --saved_model_name non_basic_aug
