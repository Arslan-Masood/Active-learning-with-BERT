#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --job-name=ClinTox_AL
#SBATCH --array=0-39  # 10 seeds * 2 strategies * 1 dataset * 2 feature types = 40 combinations
#SBATCH --output=/scratch/work/masooda1/logdir/active_learning_ClinTox_%a.out

# Directory configurations
BASE_DATA_DIR="/scratch/work/masooda1/datasets/datasets_for_active_learning"
BASE_OUTPUT_DIR="/scratch/cs/pml/AI_drug/trained_model_pred/active_learning/ClinTox"
VENV_PATH="/scratch/work/masooda1/.conda_envs/env_arslan"

# Define arrays
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

STRATEGY_ARRAY=(
    "ucb"
    "greedy"
)

FEATURES_TYPE_ARRAY=(
    "FP"
    "BERT"
)

# Single dataset
DATASET="clintox"

# Calculate array indices
NUM_SEEDS=${#SEED_ARRAY[@]}
NUM_STRATEGIES=${#STRATEGY_ARRAY[@]}
NUM_FEATURES=${#FEATURES_TYPE_ARRAY[@]}

# Debug prints
echo "Debug: SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "Debug: NUM_SEEDS = $NUM_SEEDS"
echo "Debug: NUM_STRATEGIES = $NUM_STRATEGIES"
echo "Debug: NUM_FEATURES = $NUM_FEATURES"

# Calculate indices
TOTAL_COMBINATIONS=$((NUM_STRATEGIES * NUM_SEEDS))
FEATURES_INDEX=$(( SLURM_ARRAY_TASK_ID / TOTAL_COMBINATIONS ))
REMAINDER=$(( SLURM_ARRAY_TASK_ID % TOTAL_COMBINATIONS ))
SEED_INDEX=$(( REMAINDER / NUM_STRATEGIES ))
STRATEGY_INDEX=$(( REMAINDER % NUM_STRATEGIES ))

# Get the actual values
SEED=${SEED_ARRAY[$SEED_INDEX]}
STRATEGY=${STRATEGY_ARRAY[$STRATEGY_INDEX]}
FEATURES_TYPE=${FEATURES_TYPE_ARRAY[$FEATURES_INDEX]}

# Debug prints for job parameters
echo "Debug: SEED_INDEX = $SEED_INDEX"
echo "Debug: STRATEGY_INDEX = $STRATEGY_INDEX"
echo "Debug: SEED = $SEED"
echo "Debug: STRATEGY = $STRATEGY"
echo "Debug: FEATURES_INDEX = $FEATURES_INDEX"
echo "Debug: FEATURES_TYPE = $FEATURES_TYPE"

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Error: Incorrect number of arguments. Usage: ./run_active_learning_clintox.sh <config_file>"
    exit 1
fi

echo "Step 1: Arguments provided. Proceeding with assignment."

# Assign arguments to variables
CONFIG_FILE=$1

echo "Step 2: Arguments assigned. Config file: $CONFIG_FILE"

# Create dataset-specific paths
TARGET_FILE="${BASE_DATA_DIR}/raw_data/clintox.csv"
BERT_FEATURES_FILE="${BASE_DATA_DIR}/MolBERT_features/MolBERT_clintox.csv"
ECFP_FEATURES_FILE="${BASE_DATA_DIR}/MF/MF_r2_1024_ClinTox.csv"
POS_WEIGHTS_FILE="${BASE_DATA_DIR}/raw_data/pos_ratio.csv"

METADATA_DIR="${BASE_OUTPUT_DIR}/${FEATURES_TYPE}"
WANDB_DIR="${METADATA_DIR}/wandb"

# Create required directories
mkdir -p "$METADATA_DIR"
mkdir -p "$WANDB_DIR"

# Validate that input files exist
for file in "$TARGET_FILE" "$BERT_FEATURES_FILE" "$POS_WEIGHTS_FILE"; do
    if [ ! -f "$file" ]; then
        echo "Error: Required file does not exist: $file"
        exit 1
    fi
done

# Create logs directory inside metadata_dir
LOGS_DIR="${METADATA_DIR}/logs"
mkdir -p "$LOGS_DIR"

# Generate output filename based on seed and sampling method
OUTPUT_FILE="${LOGS_DIR}/${STRATEGY}_${DATASET}_seed${SEED}_${FEATURES_TYPE}.out"
echo "Output will be saved to: $OUTPUT_FILE"

# Load mamba and activate the environment
echo "Step 3: Activating conda environment..."
module load mamba
source activate "$VENV_PATH"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment."
    exit 1
fi

echo "Step 4: Conda environment activated. Now running the Python script."

# Create temporary config file and update paths
TMP_CONFIG="/tmp/config_${SLURM_ARRAY_TASK_ID}.json"
cp "$CONFIG_FILE" "$TMP_CONFIG"

# Update config file
sed -i "s|\"target_file\": \"[^\"]*\"|\"target_file\": \"${TARGET_FILE}\"|" "$TMP_CONFIG"
sed -i "s|\"BERT_features_file\": \"[^\"]*\"|\"BERT_features_file\": \"${BERT_FEATURES_FILE}\"|" "$TMP_CONFIG"
sed -i "s|\"ECFP_features_file\": \"[^\"]*\"|\"ECFP_features_file\": \"${ECFP_FEATURES_FILE}\"|" "$TMP_CONFIG"
sed -i "s|\"pos_weights\": \"[^\"]*\"|\"pos_weights\": \"${POS_WEIGHTS_FILE}\"|" "$TMP_CONFIG"
sed -i "s|\"features_type\": \"[^\"]*\"|\"features_type\": \"${FEATURES_TYPE}\"|" "$TMP_CONFIG"

# Update input dimension based on feature type
if [ "$FEATURES_TYPE" = "BERT" ]; then
    sed -i "s|\"input_dim\": \"[^\"]*\"|\"input_dim\": 768|" "$TMP_CONFIG"
elif [ "$FEATURES_TYPE" = "FP" ]; then
    sed -i "s|\"input_dim\": \"[^\"]*\"|\"input_dim\": 1024|" "$TMP_CONFIG"
fi

# Update metadata directory to include feature type
METADATA_DIR="${BASE_OUTPUT_DIR}/${FEATURES_TYPE}"
WANDB_DIR="${METADATA_DIR}/wandb"

# Update the rest of the paths in config
sed -i "s|\"metadata_dir\": \"[^\"]*\"|\"metadata_dir\": \"${METADATA_DIR}/\"|" "$TMP_CONFIG"
sed -i "s|\"wandb_dir\": \"[^\"]*\"|\"wandb_dir\": \"${WANDB_DIR}/\"|" "$TMP_CONFIG"

# Debug prints for paths
echo "Debug: Target file path: $TARGET_FILE"
echo "Debug: BERT features file path: $BERT_FEATURES_FILE"
echo "Debug: ECFP features file path: $ECFP_FEATURES_FILE"
echo "Debug: Features type: $FEATURES_TYPE"

# Update strategy, seed, and dataset
sed -i "s/\"sampling_strategy\": \"[^\"]*\"/\"sampling_strategy\": \"$STRATEGY\"/" "$TMP_CONFIG"
sed -i "s/\"seed\": \"[^\"]*\"/\"seed\": $SEED/" "$TMP_CONFIG"
sed -i "s/\"dataset\": \"[^\"]*\"/\"dataset\": \"$DATASET\"/" "$TMP_CONFIG"

# Run the Python script with the specified seed
srun python /scratch/work/masooda1/active_learning/scripts/Active_learning.py --config_file "$TMP_CONFIG" --seed "$SEED" > "$OUTPUT_FILE"

# Check if the Python script executed successfully
if [ $? -eq 0 ]; then
    echo "Step 5: Python script executed successfully."
    
    # Wait a few seconds to ensure all file operations are complete
    sleep 5
    
    # Delete the temporary config file
    rm "$TMP_CONFIG"
    echo "Temporary config file deleted: $TMP_CONFIG"
    
    # Clean up model weights for this specific strategy and task
    WEIGHTS_DIR="${METADATA_DIR}/${STRATEGY}/Y/model_weights"
    if [ -d "$WEIGHTS_DIR" ]; then
        # Check if any processes are using files in the directory
        if ! lsof "$WEIGHTS_DIR"/* > /dev/null 2>&1; then
            echo "Cleaning up model weights directory: $WEIGHTS_DIR"
            rm -rf "$WEIGHTS_DIR"
            echo "Successfully deleted model weights directory"
        else
            echo "Warning: Model weights directory is still in use, skipping deletion"
        fi
    else
        echo "Debug: Weights directory not found at: $WEIGHTS_DIR"
    fi
    
    # Delete the log file if execution was successful
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