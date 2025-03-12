import pandas as pd
import numpy as np
import os
import argparse
from deepchem.feat import CircularFingerprint
import warnings
import logging
from rdkit import Chem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Filter the deprecation warning
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def is_valid_smiles(smiles):
    """Check if a SMILES string is valid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def read_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.parquet':
        return pd.read_parquet(file_path)
    elif file_extension.lower() == '.tab':
        return pd.read_csv(file_path, sep='\t')
    else:
        return pd.read_csv(file_path)
    
def generate_morgan_fingerprints(input_file, output_file, smiles_column='SMILES', size=1024, radius=2):
    """
    Generate Morgan Fingerprints using deepchem and save to CSV
    """
    # Read the file
    try:
        df = read_file(input_file)
    except Exception as e:
        raise ValueError(f"Error reading file {input_file}: {str(e)}")
    
    # Check if SMILES column exists
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in the input file")
    
    # Initialize the fingerprint generator
    featurizer = CircularFingerprint(size=size, radius=radius)
    
    # Create lists to store results
    valid_smiles = []
    invalid_smiles = []
    fingerprints_list = []
    
    # Process each SMILES
    for idx, smiles in enumerate(df[smiles_column]):
        if pd.isna(smiles):
            logger.warning(f"Row {idx}: Found NA SMILES")
            invalid_smiles.append((idx, smiles, "NA value"))
            continue
            
        if not isinstance(smiles, str):
            logger.warning(f"Row {idx}: SMILES is not a string: {type(smiles)}")
            invalid_smiles.append((idx, smiles, "Not a string"))
            continue
            
        if not is_valid_smiles(smiles):
            logger.warning(f"Row {idx}: Invalid SMILES: {smiles}")
            invalid_smiles.append((idx, smiles, "Invalid SMILES"))
            continue
            
        try:
            # Generate fingerprint for single SMILES
            fp = featurizer.featurize([smiles])
            if fp is not None and len(fp) > 0:
                fingerprints_list.append(fp[0])
                valid_smiles.append(smiles)
            else:
                logger.warning(f"Row {idx}: Failed to generate fingerprint for SMILES: {smiles}")
                invalid_smiles.append((idx, smiles, "Featurization failed"))
        except Exception as e:
            logger.warning(f"Row {idx}: Error processing SMILES {smiles}: {str(e)}")
            invalid_smiles.append((idx, smiles, str(e)))
    
    # Convert fingerprints to numpy array
    if fingerprints_list:
        fingerprints = np.array(fingerprints_list, dtype=np.float32)
        
        # Create column names for fingerprints
        fp_columns = [f'MF_{i}' for i in range(size)]
        
        # Create the final DataFrame
        result_df = pd.DataFrame(fingerprints, columns=fp_columns)
        result_df.insert(0, 'SMILES', valid_smiles)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the successful results
        result_df.to_csv(output_file, index=False)
        
        # Save failed SMILES to a separate file
        failed_df = pd.DataFrame(invalid_smiles, columns=['Row_Index', 'SMILES', 'Error'])
        failed_file = os.path.splitext(output_file)[0] + '_failed.csv'
        failed_df.to_csv(failed_file, index=False)
        
        # Log summary
        logger.info(f"Total SMILES processed: {len(df)}")
        logger.info(f"Successfully processed: {len(valid_smiles)}")
        logger.info(f"Failed: {len(invalid_smiles)}")
        logger.info(f"Failed SMILES saved to: {failed_file}")
        
        return fingerprints.shape
    else:
        raise ValueError("No valid fingerprints were generated")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate Morgan Fingerprints from SMILES')
    parser.add_argument('-i', '--input', required=True, help='Input CSV file path')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file path')
    parser.add_argument('-s', '--smiles_column', required=True, help='Name of SMILES column in input CSV')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Generate fingerprints
        shape = generate_morgan_fingerprints(args.input, args.output, args.smiles_column, size=2048, radius=6)
        
        print(f"Successfully processed input file: {args.input}")
        print(f"Fingerprints generated with shape: {shape}")
        print(f"Output saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()