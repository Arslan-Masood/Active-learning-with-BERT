#!/bin/bash -l
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu-debug

#SBATCH --output=/scratch/work/masooda1/script_output/active_learning_main.out

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Error: Incorrect number of arguments. Usage: ./run_active_learning_main.sh <config_file>"
    exit 1
fi

echo "Step 1: Arguments provided. Proceeding with assignment."

# Assign arguments to variables
CONFIG_FILE=$1

echo "Step 2: Arguments assigned. Config file: $CONFIG_FILE"

# Path to your conda environment
VENV_PATH="/scratch/work/masooda1/.conda_envs/env_arslan"

# Load mamba and activate the environment
echo "Step 4: Activating conda environment..."
module load mamba
source activate $VENV_PATH
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment."
    exit 1
fi

echo "Step 5: Conda environment activated. Now running the Python script."

# Run the Python script using srun
srun python /scratch/work/masooda1/active_learning/scripts/Active_learning.py --config_file "$CONFIG_FILE"
if [ $? -ne 0 ]; then
    echo "Error: Failed to run the Python script."
    exit 1
fi

echo "Step 6: Python script executed successfully."

# Deactivate the conda environment
echo "Step 7: Deactivating the conda environment..."
conda deactivate
if [ $? -ne 0 ]; then
    echo "Error: Failed to deactivate the conda environment."
    exit 1
fi

echo "Step 8: Conda environment deactivated. Script finished successfully."
