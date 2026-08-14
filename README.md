# Yield Estimation Transformer

A transformer-based model for county-level corn yield estimation using multi-temporal weather observations and static soil properties.

The model combines weekly weather time-series with static soil features to estimate corn yield in bushels per acre (`bu/acre`). This repository contains the source code for data preparation, model training, evaluation, and inference, together with the final trained checkpoint.

The trained model is also packaged separately as a Hugging Face Transformers model and has been validated for deployment as an inference service through FlexServ.

## Tags

- Crop Yield Estimation
- Digital Agriculture
- Transformers
- Multi-Temporal Modeling
- Regression
- PyTorch
- Hugging Face Transformers
- FlexServ

## License

- [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## References

### USA County Level Crop Yield Dataset

This work uses the USA County Level Crop Yield Dataset.

Please cite both the original work and the associated processing work:

```bibtex
@inproceedings{hasan2026vita,
  title     = {VITA: Variational Pretraining of Transformers for Climate-Robust Crop Yield Forecasting},
  author    = {Adib Hasan and Mardavij Roozbehani and Munther Dahleh},
  booktitle = {Proceedings of the 40th AAAI Conference on Artificial Intelligence},
  year      = {2026}
}

@article{Khaki2020CNNRNN,
  author    = {Khaki, Saeed and Wang, Liang and Archontoulis, Sotirios V.},
  title     = {A CNN-RNN Framework for Crop Yield Prediction},
  journal   = {Frontiers in Plant Science},
  volume    = {10},
  pages     = {1750},
  year      = {2020},
  doi       = {10.3389/fpls.2019.01750},
  publisher = {Frontiers Media SA}
}
```

### FlexServ

The released model has been packaged and validated for deployment through FlexServ.

FlexServ documentation:

https://zhangwei217245.github.io/FlexServ/

## Acknowledgements

This work was developed as part of the ICICLE AI Institute.

*National Science Foundation (NSF) AI Institute for Intelligent Cyberinfrastructure with Computational Learning in the Environment (ICICLE), Award OAC-2112606.*

## Issue Reporting

For questions, issues, or support, please use the GitHub repository issue tracker.

Contact:

Sarikaa Sridhar  
sridhar.86@buckeyemail.osu.edu

---

# Tutorials

## Overview

The Yield Estimation Transformer predicts county-level corn yield from multi-temporal weather observations and static soil properties.

The final model uses six weekly weather variables:

- `prcp`
- `srad`
- `swe`
- `tmax`
- `tmin`
- `vp`

The model also uses 66 static soil features.

The weather observations are represented as weekly time-series, while soil properties provide static information for each sample.

The final multi-temporal model supports the following seasonal prediction cutoffs:

```text
20, 24, 28, 32, 36, 40, 44, 48, 52
```

A cutoff determines how many weeks of weather observations are available to the model. For example, a cutoff of `20` performs an earlier-season prediction, while a cutoff of `52` performs full-season inference.

The core workflow in this repository is:

```text
Raw / prepared crop data
        ↓
Dataset preprocessing
        ↓
Weekly weather + static soil features
        ↓
Yield Estimation Transformer
        ↓
Multi-cutoff training
        ↓
Regression checkpoint
        ↓
Evaluation / inference
        ↓
Hugging Face pipeline
        ↓
FlexServ deployment
```

The GitHub repository and Hugging Face repository serve different purposes.

This repository contains the original regression-oriented training, evaluation, and inference implementation.

The Yield estimation model on Hugging Face contains the deployment-oriented package used to load the trained model through a standard Hugging Face pipeline supported by FlexServ.

## Prerequisites

The project requires:

- Python
- PyTorch
- Hugging Face Transformers
- NumPy
- Pandas
- HDF5-related dependencies used by the dataset pipeline
- Additional dependencies listed in `requirements.txt`

For GPU training, a CUDA-compatible PyTorch environment is recommended.

The included SLURM script can be used on a compatible HPC system.

---

# How-To Guides

## Problem Description

The objective is to estimate county-level corn yield from weather and soil information.

Each sample contains two primary sources of information:

1. multi-temporal weather observations;
2. static soil properties.

The weather input is represented as:

```text
[K, W]
```

where:

- `K` is the number of temporal observations;
- `W = 6` is the number of weather variables.

The static soil input is represented as:

```text
[S]
```

where:

```text
S = 66
```

The model learns a regression function of the form:

```text
weather + soil + crop information
                ↓
        transformer model
                ↓
       predicted corn yield
```

The predicted value is expressed in bushels per acre (`bu/acre`).

The model supports multi-temporal estimation by evaluating the weather sequence at different seasonal cutoffs.

## Getting Started

The final repository is organized as follows:

```text
.
├── README.md
├── requirements.txt
├── training.slurm
│
├── checkpoints/
│   ├── config.json
│   ├── metrics.json
│   └── model.safetensors
│
├── config/
│   ├── __init__.py
│   └── config.py
│
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   └── preprocessing.py
│
├── examples/
│   └── sample_input_weekly.json
│
├── hf/
│   ├── __init__.py
│   ├── auto.py
│   ├── configuration_yield.py
│   └── modeling_yield.py
│
├── models/
│   ├── __init__.py
│   └── unimodal_ws_crossattn.py
│
├── scripts/
│   ├── __init__.py
│   ├── prepare_cornbelt.py
│   ├── train_hf.py
│   ├── evaluate_hf.py
│   └── inference_hf.py
│
└── training/
    ├── __init__.py
    └── engine.py
```

The major components are:

- `data/` — dataset loading and preprocessing
- `models/` — core neural network architecture
- `training/` — training and evaluation utilities
- `hf/` — Hugging Face AutoClass-compatible regression wrapper used by the training repository
- `scripts/` — data preparation, training, evaluation, and inference entry points
- `checkpoints/` — final trained checkpoint and configuration
- `examples/` — example structured model input
- `training.slurm` — example HPC training job

## Installation

Clone the repository:

```bash
git clone https://github.com/ICICLE-ai/yield_estimation.git
cd yield_estimation
```

Create a Python environment:

```bash
conda create -n yield_hf python=3.11
conda activate yield_hf
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

The USA County Level Crop Yield public dataset is used for training, validation and testing. The data preparation workflow is implemented in:

```text
scripts/prepare_cornbelt.py
```

After preparation, the expected dataset structure is:

```text
data/
└── cornbelt/
    ├── train.h5
    ├── val.h5
    └── test.h5
```

The model uses the following six weather variables:

```text
prcp
srad
swe
tmax
tmin
vp
```

The 66 soil variables used by the final checkpoint are recorded in the model configuration.

## Training

The primary training entry point is:

```text
scripts/train_hf.py
```

The final model uses multi-cutoff training with:

```text
20,24,28,32,36,40,44,48,52
```

An example training command is:

```bash
python scripts/train_hf.py \
  --train_file data/cornbelt/train.h5 \
  --val_file data/cornbelt/val.h5 \
  --test_file data/cornbelt/test.h5 \
  --weather_vars prcp,srad,swe,tmax,tmin,vp \
  --soil_vars bdod_mean_0-5cm,bdod_mean_5-15cm,bdod_mean_15-30cm,bdod_mean_30-60cm,bdod_mean_60-100cm,bdod_mean_100-200cm,cec_mean_0-5cm,cec_mean_5-15cm,cec_mean_15-30cm,cec_mean_30-60cm,cec_mean_60-100cm,cec_mean_100-200cm,cfvo_mean_0-5cm,cfvo_mean_5-15cm,cfvo_mean_15-30cm,cfvo_mean_30-60cm,cfvo_mean_60-100cm,cfvo_mean_100-200cm,clay_mean_0-5cm,clay_mean_5-15cm,clay_mean_15-30cm,clay_mean_30-60cm,clay_mean_60-100cm,clay_mean_100-200cm,nitrogen_mean_0-5cm,nitrogen_mean_5-15cm,nitrogen_mean_15-30cm,nitrogen_mean_30-60cm,nitrogen_mean_60-100cm,nitrogen_mean_100-200cm,ocd_mean_0-5cm,ocd_mean_5-15cm,ocd_mean_15-30cm,ocd_mean_30-60cm,ocd_mean_60-100cm,ocd_mean_100-200cm,ocs_mean_0-5cm,ocs_mean_5-15cm,ocs_mean_15-30cm,ocs_mean_30-60cm,ocs_mean_60-100cm,ocs_mean_100-200cm,phh2o_mean_0-5cm,phh2o_mean_5-15cm,phh2o_mean_15-30cm,phh2o_mean_30-60cm,phh2o_mean_60-100cm,phh2o_mean_100-200cm,sand_mean_0-5cm,sand_mean_5-15cm,sand_mean_15-30cm,sand_mean_30-60cm,sand_mean_60-100cm,sand_mean_100-200cm,silt_mean_0-5cm,silt_mean_5-15cm,silt_mean_15-30cm,silt_mean_30-60cm,silt_mean_60-100cm,silt_mean_100-200cm,soc_mean_0-5cm,soc_mean_5-15cm,soc_mean_15-30cm,soc_mean_30-60cm,soc_mean_60-100cm,soc_mean_100-200cm \
  --crop corn \
  --time_agg weekly \
  --train_cutoffs 20,24,28,32,36,40,44,48,52 \
  --eval_cutoffs 20,24,28,32,36,40,44,48,52 \
  --epochs 30 \
  --lr 3e-5 \
  --batch_size 32 \
  --out_dir checkpoints
```

The final checkpoint is stored in:

```text
checkpoints/
```

The checkpoint includes:

```text
config.json
model.safetensors
metrics.json
```
## Training on Your Own Data

The training pipeline can also be used to train a new yield estimation model on a compatible dataset.

Prepare the dataset in the HDF5 format expected by `YieldDataset` and provide separate training, validation, and test files.

The weather and soil variables supplied to the training command must correspond to the variables available in the prepared dataset.

A general training command is:

```bash
python scripts/train_hf.py \
  --train_file <path/to/train.h5> \
  --val_file <path/to/val.h5> \
  --test_file <path/to/test.h5> \
  --weather_vars <comma-separated-weather-variables> \
  --soil_vars <comma-separated-soil-variables> \
  --crop <crop-name> \
  --time_agg weekly \
  --train_cutoffs <comma-separated-training-cutoffs> \
  --eval_cutoffs <comma-separated-evaluation-cutoffs> \
  --epochs <number-of-epochs> \
  --lr <learning-rate> \
  --batch_size <batch-size> \
  --out_dir <output-directory>
```

### SLURM Training

An example SLURM job is provided in:

```text
training.slurm
```

Submit it using:

```bash
sbatch training.slurm
```

## Evaluation

The trained checkpoint can be evaluated using:

```text
scripts/evaluate_hf.py
```

For the final multi-cutoff model:

```bash
python scripts/evaluate_hf.py \
  --hf_model_dir checkpoints \
  --test_file data/cornbelt/test.h5 \
  --cutoffs 20,24,28,32,36,40,44,48,52 \
  --batch_size 64 \
  --output_csv checkpoints/test_predictions.csv \
  --metrics_json checkpoints/test_metrics.json
```

Evaluation is performed independently at the configured seasonal cutoffs.

The evaluation process:

1. loads the final trained checkpoint;
2. loads the test dataset;
3. applies the normalization statistics stored in the checkpoint configuration;
4. performs inference at the requested cutoffs;
5. computes evaluation metrics;
6. save predictions and metrics to disk.

## Inference

Inference using the trained checkpoint is implemented in:

```text
scripts/inference_hf.py
```

An example structured input is provided in:

```text
examples/sample_input_weekly.json
```

The sample follows the general structure:

```json
{
  "crop": "corn",
  "weather_format": "weekly",
  "cutoff": 52,
  "weather": {
    "prcp": [],
    "srad": [],
    "swe": [],
    "tmax": [],
    "tmin": [],
    "vp": []
  },
  "soil": {
    "bdod_mean_0-5cm": 0.0
  }
}
```

The complete sample file contains the required weather sequence and soil variables.

Run single-sample inference with:

```bash
python scripts/inference_hf.py \
  --hf_model_dir checkpoints \
  --single_sample_json examples/sample_input_weekly.json \
  --cutoff 52 \
  --output_csv inference_prediction.csv
```

The output contains the predicted yield for the requested cutoff.

For example:

```text
sample_idx,cutoff,y_pred
0,52,<predicted_yield>
```

## Yield Estimation on Hugging Face

The trained checkpoint is available on Hugging Face for inference. 

The deployment flow is:

```text
Structured yield input
        ↓
JSON-formatted string
        ↓
YieldTokenizer
        ↓
weather + soil + crop + cutoff tensors
        ↓
Yield Estimation Transformer
        ↓
yield regression
        ↓
YIELD_BU_ACRE
```

The Hugging Face deployment package uses the standard task:

```text
text-classification
```

pipeline as a compatibility interface for FlexServ.

This does **not** change the underlying regression task. The model remains a regression model, and the returned `score` represents predicted corn yield in `bu/acre`.

The yield estimation inference service can be performed by using Hugging Face model through Flexserv.

## FlexServ Deployment

The Yield estimation Hugging Face model has been tested through FlexServ.

The serving configuration uses:

```text
Task:  text-classification
Model: ICICLE-AI/yield-estimation
```

FlexServ requires the `inputs` field for this task to be a string. Therefore, the structured yield input is serialized into a JSON-formatted string before being submitted to the service.

The request looks like:

```json
{
  "task": "text-classification",
  "inputs": "{\"crop\":\"corn\",\"weather_format\":\"weekly\",\"cutoff\":52,\"weather\":{...},\"soil\":{...}}",
  "parameters": {},
  "model": "ICICLE-AI/yield-estimation"
}
```

A successful response looks like:

```json
[
  {
    "label": "YIELD_BU_ACRE",
    "score": 165.1769561767578
  }
]
```

The `score` is the estimated corn yield in `bu/acre`.

---

# Explanation

## Input Features

### Weather

The final model uses six weather variables:

```text
prcp
srad
swe
tmax
tmin
vp
```

The expected weather feature dimension is:

```text
W = 6
```

### Soil

The model uses 66 static soil features.

The expected soil feature dimension is:

```text
S = 66
```

The exact soil feature names and ordering are stored in:

```text
checkpoints/config.json
```

### Crop

The final released model is intended for corn yield estimation.

### Normalization

Weather and soil inputs are normalized using statistics derived from the training data.

The normalization parameters are stored with the trained checkpoint:

```text
w_mean
w_std
s_mean
s_std
```

The output normalization parameters are also stored in the configuration:

```text
y_mean
y_std
```

This allows the trained checkpoint to preserve the preprocessing and output scaling information required for inference.

## Multi-Temporal Yield Estimation

The final model was designed for yield estimation at multiple points during the growing season.

The supported training and evaluation cutoffs are:

```text
20
24
28
32
36
40
44
48
52
```

At a given cutoff, only weather observations available up to that point are used by the temporal model.

This allows the same trained model to support both earlier-season and full-season yield estimation.

## Features

- **Transformer-Based Yield Estimation:** Models temporal weather information using a transformer architecture.
- **Weather and Soil Integration:** Combines six weekly weather variables with 66 static soil properties.
- **County-Level Corn Yield Prediction:** Produces yield estimates in bushels per acre.
- **Multi-Temporal Training:** Supports multiple seasonal prediction cutoffs from week 20 through week 52.
- **Training Reproducibility:** Includes data preparation, training, evaluation, and inference scripts.
- **Hugging Face Checkpoint Format:** Stores the final trained model using Hugging Face-compatible configuration and Safetensors weights.
- **Standalone Inference:** Supports inference from structured sample input using the original regression checkpoint.
- **HPC Training:** Includes an example SLURM training workflow.
- **Hugging Face model:** Provides a separately packaged pretrained model for convenient inference.
- **FlexServ Deployment:** The released Hugging Face package can be used to perform inference through a FlexServ-supported inference interface.
- **CPU and GPU Support:** Supports PyTorch inference and training on compatible CPU and CUDA environments.
