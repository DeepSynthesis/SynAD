# SynAD: Synthesis Applicability Domain

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SynAD (Synthesis Applicability Domain) is a Python package for evaluating the synthesis applicability domain of chemical reaction prediction models. It provides tools for training machine learning models, evaluating their applicability domains, and computing confidence scores for reaction predictions.

## Features

- **Model Training**: Train various machine learning models (XGBoost, LightGBM, CatBoost, Neural Networks, etc.) for chemical reaction prediction
- **SynAD Evaluation**: Evaluate the synthesis applicability domain of trained models
- **SynAD Score Computation**: Calculate confidence scores for reaction predictions
- **Multiple Datasets**: Support for various chemical reaction datasets
- **Flexible Architecture**: Modular design for easy extension and customization

## Installation

### From Source (Development)

```bash
git clone https://github.com/deepsynthesis/synad.git
cd synad
conda create -n synad python==3.10.3
conda activate synad
pip install -e .
```

### Development Installation with Optional Dependencies

```bash
git clone https://github.com/deepsynthesis/synad.git
cd synad
conda create -n synad-dev python==3.10.3
conda activate synad
pip install -e ".[dev]"
```


### System Requirements

- Python 3.10 or higher
- See `pyproject.toml` for complete dependency list

## Quick Start

### Basic Usage

see [`demo.ipynb`](demo.ipynb) for a detailed example of how to use SynAD for training, evaluating, and scoring reactions.

### SynAD evaluation with ULD

run `python src/synad_eval_for_ULD.py`

### SynAD evaluation with other datasets

run `python src/synad_eval_for_datasets.py`


## Core Components

### 1. Model Training (`synad.models`)
- Support for multiple ML algorithms (tree-based, neural networks)
- Automated hyperparameter optimization
- Cross-validation and evaluation metrics

### 2. SynAD Evaluation (`synad.evaluation`)
- `SynADJudgementor`: Main class for applicability domain evaluation
- `SynADScoreEvaluator`: Confidence score computation
- Multiple distance metrics and evaluation strategies

### 4. Utilities (`util_func`)
- Helper functions for data processing
- Visualization tools
- Performance metrics

## Supported Models

SynAD is model-agnostic, supporting a wide range of models from classical machine learning algorithms (e.g., XGBoost, Random Forest, SVM) to complex neural networks.

## Supported Datasets

- ULD (Ullmann ligand dataset).
- Custom reaction datasets (with proper formattings)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

