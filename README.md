# Active Learning with BERT for Molecular Property Prediction

This repository contains the implementation of our paper "Molecular Property Prediction using Pretrained-BERT and Bayesian Active Learning: A Data-Efficient Approach to Drug Design". 

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Citation](#citation)
- [Contact](#contact)

## Overview

Our framework achieves efficient molecular property prediction by:
- Leveraging pretrained BERT representations in active learning framework
- Using Bayesian acquisition functions (BALD, EPIG) for active learning
- Demonstrating effectiveness on toxicity and ADME property prediction

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Arslan-Masood/Active-learning-with-BERT.git
cd Active-learning-with-BERT
```

2. Create and activate a conda environment:

```bash
conda create -y -q -n ActiveBERT python=3.9.10
conda activate ActiveBERT
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data

### Datasets
We use three benchmark datasets:

1. **Toxicity Datasets**:
   - **Tox21**: Toxicity prediction dataset with 12 different toxicity endpoints
   - **ClinTox**: Clinical toxicity dataset focusing on drug safety

2. **ADME Dataset**:
   - 2 classification datasets from TDC-ADME benchmark:
      - PAMPA Permeability, NCATS
      - Pgp (P-glycoprotein) Inhibition, Broccatelli et al.

### Download Instructions
1. Download the complete `datasets_for_active_learning` folder from [Figshare](https://figshare.com/articles/dataset/Datasets_and_computed_features/28580027)
2. This folder contains:
   - Raw molecular data
   - Precomputed BERT features (using [MolBERT](https://github.com/BenevolentAI/MolBERT))
   - Computed Morgan fingerprints
3. Place the downloaded data in the `datasets` directory

## Usage

### Running Experiments

#### Tox21 Dataset
With BERT Features:
```bash
sbatch scripts/Active_learning_Tox21.sh configs/Tox21/BERT/Tox21_BERT.json
```

With Morgan Fingerprints (ECFP):
```bash
sbatch scripts/Active_learning_Tox21.sh configs/Tox21/MF/Tox21_MF.json
```

#### ClinTox Dataset
With BERT Features:
```bash
sbatch scripts/Active_learning.sh configs/clintox/MolBERT_features/ClinTox_BALD.json
```

With Morgan Fingerprints (ECFP):
```bash
sbatch scripts/Active_learning.sh configs/clintox/Morg_FP_features/ClinTox_BALD.json
```

#### ADME Properties
With BERT and ECFP Features:
```bash
sbatch /scripts/Ative_learning_ADME.sh /scripts/configs/ADME/ADME.json
```

## Citation

If you use this code in your research, please cite:
```bibtex
@article{masood2024molecular,
    title={Molecular Property Prediction using Pretrained-BERT and Bayesian Active Learning: A Data-Efficient Approach to Drug Design},
    author={Muhammad Arslan Masood},
    journal={under review},
    year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Muhammad Arslan Masood**
- Email: arslan.masood@aalto.fi
- Institution: Aalto University
