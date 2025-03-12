#!/bin/bash -l
#SBATCH --time=120:00:00
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --job-name=Tox21_BERT
#SBATCH --array=0-107  # 12 tasks * 3 seeds * 3 strategies = 108 total combinations
#SBATCH --output=/scratch/work/masooda1/trials/active_learning_Tox21_ECFP_%a.out

# Define arrays
SEED_ARRAY=(
   0    # Seed for the first job
   42   # Seed for the second job
   88   # Seed for the third job
)

TASK_ARRAY=(
    "NR-AR"
    "NR-AR-LBD"
    "NR-AhR"
    "NR-Aromatase"
    "NR-ER"
    "NR-ER-LBD"
    "NR-PPAR-gamma"
    "SR-ARE"
    "SR-ATAD5"
    "SR-HSE"
    "SR-MMP"
    "SR-p53"
)

STRATEGY_ARRAY=(
    "BALD"
    "EPIG_MT"
    "uniform"
)

# Calculate array indices
NUM_SEEDS=${#SEED_ARRAY[@]}
NUM_TASKS=${#TASK_ARRAY[@]}
NUM_STRATEGIES=${#STRATEGY_ARRAY[@]}

# Calculate indices
SEED_INDEX=$(( SLURM_ARRAY_TASK_ID / (NUM_TASKS * NUM_STRATEGIES) ))
REMAINING=$(( SLURM_ARRAY_TASK_ID % (NUM_TASKS * NUM_STRATEGIES) ))
TASK_INDEX=$(( REMAINING / NUM_STRATEGIES ))
STRATEGY_INDEX=$(( REMAINING % NUM_STRATEGIES ))

# Get the actual values
SEED=${SEED_ARRAY[$SEED_INDEX]}
TASK=${TASK_ARRAY[$TASK_INDEX]}
STRATEGY=${STRATEGY_ARRAY[$STRATEGY_INDEX]}

# Debug prints
echo "Debug: SEED = $SEED"
echo "Debug: TASK = $TASK"
echo "Debug: STRATEGY = $STRATEGY"

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

# Add debug prints to see what was extracted
echo "Debug: Extracted SAMPLING_METHOD = $SAMPLING_METHOD"
echo "Debug: Extracted METADATA_DIR = $METADATA_DIR"
if [ -z "$SAMPLING_METHOD" ] || [ -z "$METADATA_DIR" ]; then
    echo "Error: Sampling method or metadata_dir not found in config file."
    exit 1
fi

# Create logs directory inside metadata_dir if it doesn't exist
LOGS_DIR="${METADATA_DIR}/logs"
mkdir -p "$LOGS_DIR"

# Generate output filename based on seed and sampling method
OUTPUT_FILE="${LOGS_DIR}/${STRATEGY}_${TASK}_seed${SEED}_${SLURM_ARRAY_TASK_ID}.out"
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

# Modify the config file with the selected task, strategy, and project name
TMP_CONFIG="/tmp/config_${SLURM_ARRAY_TASK_ID}.json"
cp "$CONFIG_FILE" "$TMP_CONFIG"

# Update main_task, sampling_strategy, and project_name
sed -i "s/\"main_task\": \"[^\"]*\"/\"main_task\": \"$TASK\"/" "$TMP_CONFIG"
sed -i "s/\"sampling_strategy\": \"[^\"]*\"/\"sampling_strategy\": \"$STRATEGY\"/" "$TMP_CONFIG"
sed -i "s/\"project_name\": \"[^\"]*\"/\"project_name\": \"${TASK}_${STRATEGY}\"/" "$TMP_CONFIG"

# Update CONFIG_FILE to use the temporary version
CONFIG_FILE=$TMP_CONFIG

# Run the Python script with the specified seed
srun python /scratch/work/masooda1/active_learning/scripts/Active_learning.py --config_file "$CONFIG_FILE" --seed "$SEED" > "$OUTPUT_FILE"

# Check if the Python script executed successfully
if [ $? -eq 0 ]; then
    echo "Step 5: Python script executed successfully."
    
    # Wait a few seconds to ensure all file operations are complete
    sleep 5
    
    # Clean up model weights for this specific task and strategy
    WEIGHTS_DIR="${METADATA_DIR}/${STRATEGY}/${TASK}/model_weights"
    if [ -d "$WEIGHTS_DIR" ]; then
        # Check if any processes are using files in the directory
        if ! lsof "$WEIGHTS_DIR"/* > /dev/null 2>&1; then
            echo "Cleaning up model weights directory: $WEIGHTS_DIR"
            #rm -rf "$WEIGHTS_DIR"
            echo "Successfully deleted model weights directory"
        else
            echo "Warning: Model weights directory is still in use, skipping deletion"
        fi
    fi
    
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