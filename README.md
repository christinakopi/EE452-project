# EE452-project - Graph-Based EEG Seizure Detection
EE-452 Network Machine Learning project

This project explores the use of Graph Neural Networks (GNNs) for seizure detection from multichannel EEG data. Traditional time-series models often overlook the spatial relationships among EEG electrodes. By representing electrodes as nodes in a graph and defining edges based on physical proximity or functional connectivity, GNNs can aggregate localized information and improve classification performance.

## Objectives

- Capture spatial and functional relationships in EEG data using graph representations.
- Benchmark different GNN architectures for binary seizure classification.
- Evaluate performance using cross-validation and a final held-out Kaggle test set.

## Graph Representations

EEG channels are modeled as nodes in a graph. Edges are constructed using:
- Physical adjacency (based on electrode placement)
- Functional relationships (e.g., Pearson correlation between signals)

## GNN Architectures Used

The following models are implemented and evaluated:
- **GCN (Graph Convolutional Network)**
- **GAT (Graph Attention Network)**
- **GIN (Graph Isomorphism Network)**
- **GraphSAGE**

## Dataset

- Multichannel EEG data for seizure detection.
- Labels: Binary (seizure / non-seizure).
- Data is split into training and test set, after preprocessing and feature extraction.

## Training & Evaluation

- Input: EEG segments converted to graphs.
- Output: Binary seizure prediction.
- Validation: Cross-validation.
- Final performance evaluated on a Kaggle test set.

```
## Project Structure
├── models/               # GNN model definitions (e.g., GCN, GAT, SAGE, GIN)
├── utils/                # Utilities for graph construction and preprocessing
├── cross_validation.py   # Functions for performing cross-validation
├── filters.py            # Graph filter implementations
├── train.py              # Training and evaluation logic (e.g., train_epoch, evaluate)
├── run_pipeline.ipynb    # Jupyter notebook for executing the full training pipeline
└── README.md             # Project overview and setup instructions
```

## Requirements

- torch torchvision torchaudio
- torch-geometric numpy scipy mne pandas scikit-learn
- mne
- seiz_eeg
- PyWavelets


