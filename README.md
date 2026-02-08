# STFL-EC

This repository hosts the official implementation for the paper: **"A Practical Framework for Secure and Traceable Federated Learning in Edge Computing Scenarios".**

> **Note:** The source code and checkpoints are currently under final cleanup and organization. They will be released immediately upon the acceptance of the paper.

## Table of Contents
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Usage](#-usage)

## Project Structure
```text
STFL-EC/
├── data/               # Data preprocessing scripts
├── models/             # Model definitions (ResNet, VGG, etc.)
├── defense/            # Watermarking & Traitor Tracing logic
│   ├── embedding.py
│   ├── extraction.py
│   └── tracing.py
├── utils/              # Utility functions
├── main.py             # Entry point for training
├── config.yaml         # Configuration file
└── requirements.txt    # Dependencies
```

## Requirements

To prepare the environment, please install the dependencies:

Bash

```
pip install -r requirements.txt
```

**Main Dependencies:**

- Python >= 3.8
- PyTorch >= 1.10
- Numpy
- Scikit-learn

## Usage

Once released, you can run the traitor tracing framework with the following command:

Bash

```
# Example: Train and embed watermark
python main.py --mode train --dataset cifar10 --watermark_len 256

# Example: Trace a traitor from a suspicious model
python main.py --mode trace --model_path ./checkpoints/suspicious_model.pth
```