# Yield Estimation Deployment

A transformer-based implementation of yield estimation model using weather and soil information. The repository provides an end-to-end pipeline for data preprocessing, model training, evaluation, and inference, with native Hugging Face AutoClass support for standardized model loading and deployment.

**Tags:** Digital-Agriculture

---

## License

This project is licensed under the Apache License 2.0.

---

# Project Overview

This repository implements a transformer-based yield estimation model that predicts crop yield using weather time-series and static soil properties. The codebase is designed for both research and production deployment, supporting training from scratch, evaluation, inference, and deployment through Hugging Face AutoClass and ML workflow orchestration platforms.

## Features

* Transformer-based yield estimation model
* Weather + soil feature integration
* Configurable weather and soil variables
* Automatic preprocessing and normalization
* Train / validation / test workflows
* Multi-cutoff evaluation
* Batch inference from HDF5 datasets
* Single-sample inference from JSON
* Hugging Face AutoClass compatible
* CPU and GPU support
* Deployment-ready model packaging

---

# Tutorials

## Installation

Clone the repository

```bash
git clone <repository-url>
cd yield_estimation_deployment
```

Create a Python environment

```bash
conda create -n yield_hf python=3.10
conda activate yield_hf
```

Install dependencies

```bash
pip install -r requirements.txt
```

Verify the installation

```bash
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
```

---

## Environment

The repository automatically detects whether a GPU is available.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

No code modifications are required when switching between CPU and GPU environments.

---

## Quick Start

Train a model

```bash
python scripts/train_hf.py \
    --train_file data/train.h5 \
    --val_file data/val.h5 \
    --test_file data/test.h5 \
    --weather_vars <weather_feature_names> \
    --soil_vars <soil_feature_names> \
    --epochs 30 \
    --batch_size 16 \
    --out_dir outputs/yield_model
```

Evaluate the trained model

```bash
python scripts/evaluate_hf.py \
    --hf_model_dir outputs/yield_model \
    --test_file data/test.h5 \
    --cutoffs 4,8,12,16,22
```

Run inference

```bash
python scripts/inference_hf.py \
    --hf_model_dir outputs/yield_model \
    --input_file data/test_no_yield.h5 \
    --cutoff 16 \
    --output_csv predictions.csv
```

---

# How-To Guide

## Input Data Format

The repository accepts HDF5 datasets for training, evaluation, and batch inference.

Each row should represent a single sample.

### Feature Columns

Weather and soil variables are **user configurable**.

The variables passed through

```text
--weather_vars
--soil_vars
```

must exist as columns in the supplied HDF5 file.

This allows the model to be trained on different weather and soil feature sets without modifying the source code.

### Required Metadata Columns

The following metadata columns are expected.

| Column     | Required                 | Description                              |
| ---------- | ------------------------ | ---------------------------------------- |
| crop       | Yes                      | Crop identifier                          |
| farm_field | Yes                      | Field identifier used for data splitting |
| year       | Yes                      | Growing season                           |
| yield      | Training/Evaluation only | Ground truth yield                       |

During inference, the `yield` column is optional.

---

## Data Preprocessing

The preprocessing pipeline performs:

* Missing value handling
* Daily-to-weekly weather aggregation (when required)
* Feature normalization
* Log transformation of yield during training
* Automatic restoration of normalization statistics during inference

Normalization statistics computed during training are saved with the trained model and reused automatically during evaluation and inference.

---

## Training

The training script trains a model from scratch and automatically performs validation and testing.

### Required Inputs

| Argument         | Description             |
| ---------------- | ----------------------- |
| `--train_file`   | Training HDF5 file      |
| `--val_file`     | Validation HDF5 file    |
| `--test_file`    | Test HDF5 file          |
| `--weather_vars` | Weather feature columns |
| `--soil_vars`    | Soil feature columns    |

### Common Optional Arguments

| Argument       | Description      |
| -------------- | ---------------- |
| `--crop`       | Crop filter      |
| `--epochs`     | Number of epochs |
| `--batch_size` | Batch size       |
| `--lr`         | Learning rate    |
| `--out_dir`    | Output directory |

Example

```bash
python scripts/train_hf.py \
    --train_file data/train.h5 \
    --val_file data/val.h5 \
    --test_file data/test.h5 \
    --weather_vars <weather_features> \
    --soil_vars <soil_features> \
    --epochs 30 \
    --batch_size 16 \
    --out_dir outputs/yield_model
```

---

## Evaluation

The evaluation script evaluates any compatible Hugging Face model.

This includes

* models trained using this repository
* pretrained models provided with the repository

Evaluation can be performed for one or multiple temporal cutoffs.

Example

```bash
python scripts/evaluate_hf.py \
    --hf_model_dir outputs/yield_model \
    --test_file data/test.h5 \
    --cutoffs 4,8,12,16,22
```

Evaluation reports

* RMSE
* Bias
* R²
* Prediction CSV

---

## Batch Inference

Run inference on an HDF5 dataset.

Example

```bash
python scripts/inference_hf.py \
    --hf_model_dir outputs/yield_model \
    --input_file inference_dataset.h5 \
    --cutoff 16 \
    --output_csv predictions.csv
```

Supported inputs include

* HDF5 datasets with yield labels
* HDF5 datasets without yield labels
* Weekly weather inputs
* Daily weather inputs (automatically aggregated)

---

## Single-Sample Inference

Predict yield for a single sample stored in JSON format.

Example

```bash
python scripts/inference_hf.py \
    --hf_model_dir outputs/yield_model \
    --single_sample_json sample.json \
    --cutoff 16 \
    --output_csv prediction.csv
```

The script automatically

* loads preprocessing statistics
* normalizes the inputs
* performs inference
* outputs the predicted yield

---

# Explanation

## Model Architecture

The model predicts crop yield using

* Weather time-series
* Static soil properties
* Crop embedding
* Transformer-based cross-attention encoder

---

## Hugging Face AutoClass

The repository uses Hugging Face AutoClass as the primary model interface.

Training initializes the model directly through the Hugging Face configuration.

```python
from transformers import AutoConfig, AutoModel

config = AutoConfig.for_model("yield-weather-soil")
model = AutoModel.from_config(config)
```

After training

```python
model.save_pretrained(output_dir)
```

The exported model can later be loaded using

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(output_dir)
```

No checkpoint conversion is required.

---

# Expected Outputs

Training

```text
outputs/
│
├── config.json
├── model.safetensors
├── metrics.json
├── test_metrics.json
└── test_predictions.csv
```

Evaluation

```text
eval_metrics.json
eval_predictions.csv
```

Inference

```text
predictions.csv
single_sample_prediction.csv
```

---

# Acknowledgements

This work was developed as part of the ICICLE AI Institute.

*National Science Foundation (NSF) funded AI Institute for Intelligent Cyberinfrastructure with Computational Learning in the Environment (ICICLE) (OAC 2112606).*
