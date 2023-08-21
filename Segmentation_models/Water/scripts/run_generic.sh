#!/bin/bash

if [ -z "$1" ]; then
  echo "Error: Please provide a name as an argument."
  echo "Usage: .scripts/run_generic.sh <NAME>"
  exit 1
fi

NAME=$1

cat <<EOL > scripts/run_script_$NAME.sh
#!/bin/bash
#SBATCH --job-name=$NAME
#SBATCH --partition=studentkillable
#SBATCH --gpus=4
#SBATCH --time=1440
#SBATCH --mem=64G
#SBATCH --output=./logs/$NAME.out # redirect stdout
#SBATCH --error=./logs/$NAME.err  # redirect stderr

module load cuda/12.0
source activate /home/yandex/DLW2023b/dimakisilev/myenv
python src/main.py --batch_size 4  --num_epochs 100 --steps_per_epoch 550 --root_path './data' --saved_model_name $NAME
EOL

sbatch scripts/run_script_$NAME.sh
