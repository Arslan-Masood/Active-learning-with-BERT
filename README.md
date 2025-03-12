# Active Learning with BERT for Molecular Property Prediction

This repository contains the implementation of our paper "Molecular Property Prediction using Pretrained-BERT and Bayesian Active Learning: A Data-Efficient Approach to Drug Design". Our approach combines pretrained BERT with Bayesian Active Learning to achieve efficient molecular property prediction with 50% fewer labeled compounds.

<p align="center">
  [Architecture diagram showing BERT + Active Learning pipeline]
</p>

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)
- [Contact](#contact)

## Overview

Our framework achieves efficient molecular property prediction by:
- Leveraging pretrained BERT representations from 1.26M compounds
- Using Bayesian uncertainty estimation for active learning
- Achieving equivalent performance with 50% fewer labeled compounds
- Demonstrating improved uncertainty calibration

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
We use two benchmark datasets for toxicity prediction:
- **Tox21**: Toxicity prediction dataset
- **ClinTox**: Clinical toxicity dataset

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
