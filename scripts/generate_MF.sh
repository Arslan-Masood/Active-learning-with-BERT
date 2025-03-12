#!/bin/bash -l
#SBATCH --time=04:00:00
#SBATCH --mem=40G
#SBATCH --job-name=MF_gen
#SBATCH --array=0-11
#SBATCH --output=/scratch/work/masooda1/logs/MF_generation/MF_gen.out

# Path to your conda environment
VENV_PATH="/scratch/work/masooda1/.conda_envs/env_arslan"

# Load mamba and activate the environment
echo "Step 1: Activating conda environment..."
module load mamba
source activate "$VENV_PATH"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment."
    exit 1
fi

# Define input and output directories
INPUT_DIR="/scratch/work/masooda1/datasets/datasets_for_active_learning/raw_data/TDC_ADME"
OUTPUT_DIR="/scratch/work/masooda1/datasets/datasets_for_active_learning/MF"
LOG_DIR="/scratch/work/masooda1/logs/MF_generation"

# Create output and log directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# First, let's get the actual dataset names from the directory
INPUT_DIR="/scratch/work/masooda1/datasets/datasets_for_active_learning/raw_data/TDC_ADME"

# Get list of .tab files and remove the .tab extension
datasets=($(ls ${INPUT_DIR}/*.tab | xargs -n 1 basename | sed 's/\.tab$//'))

# Print the datasets for verification
echo "Found datasets:"
for dataset in "${datasets[@]}"; do
    echo "$dataset"
done

# Get the current dataset based on SLURM_ARRAY_TASK_ID
dataset="${datasets[$SLURM_ARRAY_TASK_ID]}"

input_file="${INPUT_DIR}/${dataset}.tab"
output_file="${OUTPUT_DIR}/MF_r2_1024_${dataset}.csv"

echo "Processing dataset: $dataset"
echo "Input file: $input_file"
echo "Output file: $output_file"

# Run the MF generation script
python /scratch/work/masooda1/active_learning/scripts/generate_MF.py -i "$input_file" -o "$output_file" -s Drug

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "Successfully processed $dataset"
else
    echo "Error processing $dataset"
    exit 1
fi