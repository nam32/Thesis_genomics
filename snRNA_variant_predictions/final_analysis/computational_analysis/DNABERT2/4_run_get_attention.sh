#!/usr/bin/env bash
#
# SLURM array: extract DNABERT-2 hidden activations in row chunks.
# One task processes [START:END) where END is exclusive.
#
# Tips:
# - We auto-compute TOTAL rows (excluding CSV header) unless overridden.
# - Last chunk is capped to TOTAL and tasks beyond needed range exit quickly.
# - Existing outputs are skipped so you can requeue partial runs safely.

#SBATCH --job-name=dnabert2_array
#SBATCH --output=logs/dnabert2_hidden_%A_%a.out
#SBATCH --error=logs/dnabert2_hidden_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-44
# #SBATCH --partition=gpu
# #SBATCH --gpus=1

mkdir -p logs

# Adjust chunk size
CHUNK_SIZE=1000
START=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END=$((START + CHUNK_SIZE))

# Final task: cap at total lines
TOTAL=44833
if [ $END -gt $TOTAL ]; then END=$TOTAL; fi

# Activate your environment
source ~/.bashrc
conda activate dnabert2_env

python get_hidden_activations_array.py \
  --start $START \
  --end $END \
  --input /storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/final_training_data/human_tf_all_train.csv \
  --output /storage/homefs/tj23t050/BERT/human_promoter_nam/dataset_2/prediction_results/hidden_output_part_${SLURM_ARRAY_TASK_ID}.csv \
  --model_path /storage/homefs/tj23t050/BERT/dnabert2_custom_data_promoter_2
