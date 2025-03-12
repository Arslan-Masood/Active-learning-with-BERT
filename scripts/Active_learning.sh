#!/bin/bash -l
#SBATCH --time=120:00:00
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --job-name=BERT
#SBATCH --array=0-9  # Adjust this based on the number of seeds
#SBATCH --output=/scratch/work/masooda1/trials/ClinTox_BERT_Random_%a.out


# Define an array of seeds
SEED_ARRAY=(
   1    # Seed for the first job
   2    # Seed for the second job
   3    # Seed for the third job
   4    # Seed for the fourth job
   5    # Seed for the fifth job
   6    # Seed for the sixth job
   7    # Seed for the seventh job
   8    # Seed for the eighth job
   9    # Seed for the ninth job
   10   # Seed for the tenth job
)

# Get the seed based on the SLURM_ARRAY_TASK_ID
SEED=${SEED_ARRAY[$SLURM_ARRAY_TASK_ID]}

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Error: Incorrect number of arguments. Usage: ./run_active_learning_main.sh <config_file>"
    exit 1
fi

echo "Step 1: Arguments provided. Proceeding with assignment."

# Assign arguments to variables
CONFIG_FILE=$1

echo "Step 2: Arguments assigned. Config file: $CONFIG_FILE"

# Extract the sampling method and metadata_dir from the config file
SAMPLING_METHOD=$(grep -oP '"sampling_strategy": *"\K[^"]+' "$CONFIG_FILE")
METADATA_DIR=$(grep -oP '"metadata_dir": *"\K[^"]+' "$CONFIG_FILE")

if [ -z "$SAMPLING_METHOD" ] || [ -z "$METADATA_DIR" ]; then
    echo "Error: Sampling method or metadata_dir not found in config file."
    exit 1
fi

# Create logs directory inside metadata_dir if it doesn't exist
LOGS_DIR="${METADATA_DIR%/}/logs"
mkdir -p "$LOGS_DIR"

# Generate output filename based on seed and sampling method
OUTPUT_FILE="${LOGS_DIR}/${SAMPLING_METHOD}_seed${SEED}.out"
echo "Output will be saved to: $OUTPUT_FILE"

# Path to your conda environment
VENV_PATH="/scratch/work/masooda1/.conda_envs/env_arslan"

# Load mamba and activate the environment
echo "Step 3: Activating conda environment..."
module load mamba
source activate "$VENV_PATH"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment."
    exit 1
fi

echo "Step 4: Conda environment activated. Now running the Python script."

# Run the Python script with the specified seed
srun python /scratch/work/masooda1/active_learning/scripts/Active_learning.py --config_file "$CONFIG_FILE" --seed "$SEED" > "$OUTPUT_FILE"


if [ $? -eq 0 ]; then
    echo "Step 5: Python script executed successfully."
    
    # Delete the log file
    if [ -f "$OUTPUT_FILE" ]; then
        rm "$OUTPUT_FILE"
        echo "Log file deleted: $OUTPUT_FILE"
    fi
else
    echo "Error: Python script execution failed."
    exit 1
fi
# Deactivate the conda environment
echo "Step 6: Deactivating the conda environment..."
conda deactivate
if [ $? -ne 0 ]; then
    echo "Error: Failed to deactivate the conda environment."
    exit 1
fi

echo "Step 7: Conda environment deactivated. Script finished successfully."


