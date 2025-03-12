# Active Learning with BERT for Molecular Property Prediction

This repository contains the implementation of our approach combining pretrained BERT with Bayesian Active Learning for efficient molecular property prediction in drug discovery.

<p align="center">
  [You can add your architecture diagram here]
</p>

## Table of Contents
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Active Learning Pipeline](#active-learning-pipeline)
- [Reproducing Results](#reproducing-results)
- [Results](#results)
- [Citation](#citation)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Arslan-Masood/Active-learning-with-BERT.git
cd Active-learning-with-BERT
```

2. Create and activate a conda environment:

```bash
conda create -y -q -n ActiveBERT python=3.7.3
conda activate ActiveBERT
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

The model uses two main datasets:
1. Tox21: Toxicity prediction dataset
2. ClinTox: Clinical toxicity dataset

Download the the complete folder, "datasets_for_active_learning", from the following link:
that contains the raw data, BERT features, and computed Morgan fingerprints.
We used MolBERT (https://github.com/BenevolentAI/MolBERT) to compute the BERT features.

## Active Learning Pipeline

1. All scripts and configuration files are in `scripts/`

## Reproducing Results

To reproduce our experimental results:

1. Download the datasets and compy in the folder `datasets`

For Tox21
--> with BERT Features
```bash
sbatch scripts/Active_learning_Tox21.sh configs/Tox21/BERT/Tox21_BERT.json
```
--> with ECFP
```bash
sbatch scripts/Active_learning_Tox21.sh configs/Tox21/MF/Tox21_MF.json
```

For ClinTox:
--> with BERT Features
```bash
sbatch sbatch scripts/Active_learning.sh configs/clintox/MolBERT_features/ClinTox_BALD.json
```
--> with ECFP
```bash
sbatch scripts/Active_learning.sh configs/clintox/Morg_FP_features/ClinTox_BALD.json
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Muhammad Arslan Masood - arslan.masood@aalto.fi
