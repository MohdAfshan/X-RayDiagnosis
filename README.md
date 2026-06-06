# Chest X-Ray Diagnosis

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-compatible-FF6F61)](https://pytorch.org/vision/stable/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Gradio](https://img.shields.io/badge/Gradio-5.x-F97316?logo=gradio&logoColor=white)](https://www.gradio.app/)
[![License](https://img.shields.io/badge/License-To%20Be%20Added-lightgrey)](LICENSE)

## Table of Contents

- [Project Title](#project-title)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints-if-applicable)
- [Screenshots](#screenshots)
- [Model Architecture](#model-architecture-if-machine-learning-project)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Project Title

Chest X-Ray Diagnosis is a medical imaging machine learning project that classifies chest X-ray scans into `NORMAL` and `PNEUMONIA` categories using transfer learning with a DenseNet121 backbone. The project includes a reproducible training pipeline, evaluation scripts, Grad-CAM explainability, and an interactive Streamlit application for inference.

The repository is structured to support both research workflows and presentation-quality inference. It saves training checkpoints, plots, and evaluation metrics under `outputs/`, making it easy to inspect model behavior and results after training.

## Features

- Binary chest X-ray classification for `NORMAL` vs `PNEUMONIA`.
- Transfer learning with a pretrained DenseNet121 backbone.
- Optional ResNet50 support in the model builder.
- Two-phase training: frozen backbone first, then full-network fine-tuning.
- Training resumption from the last checkpoint.
- Early stopping and learning-rate scheduling based on validation AUC.
- Class-weighted loss and label smoothing for imbalanced data.
- Data augmentation for the training split.
- Test-set evaluation with accuracy, precision, recall, F1, AUC-ROC, sensitivity, specificity, and per-class metrics.
- Grad-CAM explainability for visual inspection of model attention.
- Streamlit inference app with upload, example images, confidence display, probability bars, and heatmap controls.
- Saved artifacts for checkpoints, plots, history, and test metrics.

## Tech Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- OpenCV
- Pillow
- tqdm
- PyYAML
- Albumentations 1.3.1
- Streamlit
- Gradio

No database is used. No external API keys are required.

## Project Structure

```text
XRay-Diagnosis/
├── app/
│   └── app.py                  # Streamlit inference application
├── config.yaml                 # Data, model, training, and path settings
├── data/
│   └── raw/chest_xray/         # Train/val/test image folders
├── outputs/
│   ├── checkpoints/            # best_model.pth, last_model.pth
│   ├── plots/                  # Training and evaluation plots
│   └── results/                # JSON metrics and training history
├── requirements.txt            # Python dependencies
├── setup.py                    # Project scaffold helper
└── src/
    ├── dataset.py             # Dataset, transforms, dataloaders, class weights
    ├── evaluate.py            # Test-set evaluation and metric plotting
    ├── gradcam.py             # Grad-CAM implementation and visualizations
    ├── model.py               # Model builder, optimizer, parameter stats
    ├── train.py               # Two-phase training loop and checkpointing
    └── utils.py               # Config loading, seeding, device helpers
```

### Folder Notes

- `data/raw/chest_xray/` contains the dataset organized into `train`, `val`, and `test` splits, each with `NORMAL` and `PNEUMONIA` subfolders.
- `outputs/checkpoints/` stores model checkpoints such as `best_model.pth` and `last_model.pth`.
- `outputs/plots/` stores generated figures including training curves, confusion matrix, ROC curve, PR curve, per-class metrics, sample batches, and Grad-CAM visualizations.
- `outputs/results/` stores JSON artifacts such as `history.json` and `test_metrics.json`.

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd XRay-Diagnosis
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify the dataset layout

Confirm that the chest X-ray data is available at:

```text
data/raw/chest_xray/
```

The expected split structure is:

```text
train/NORMAL
train/PNEUMONIA
val/NORMAL
val/PNEUMONIA
test/NORMAL
test/PNEUMONIA
```

## Configuration

All runtime settings are controlled from `config.yaml`.

### Data settings

- `data.data_dir`: Path to the dataset root. Default: `data/raw/chest_xray`
- `data.img_size`: Input image size. Default: `224`
- `data.batch_size`: Batch size for loaders. Default: `32`
- `data.num_workers`: DataLoader worker count. Default: `2`
- `data.num_classes`: Number of output classes. Default: `2`
- `data.class_names`: Class order used throughout the project. Default: `["NORMAL", "PNEUMONIA"]`

### Model settings

- `model.architecture`: Model backbone. Default: `densenet121`
- `model.pretrained`: Whether to load pretrained ImageNet weights. Default: `true`
- `model.dropout`: Dropout used in the classification head. Default: `0.5`
- `model.freeze_backbone`: Whether to freeze the backbone at the start of training. Default: `true`

### Training settings

- `training.epochs`: Total number of epochs. Default: `25`
- `training.learning_rate`: Base learning rate. Default: `0.0001`
- `training.weight_decay`: Weight decay for Adam. Default: `1e-4`
- `training.lr_patience`: Validation patience for `ReduceLROnPlateau`. Default: `3`
- `training.lr_factor`: Learning-rate reduction factor. Default: `0.5`
- `training.early_stop_patience`: Early stopping patience. Default: `7`
- `training.use_class_weights`: Enables class-weighted loss. Default: `true`
- `training.label_smoothing`: Label smoothing applied to cross-entropy loss. Default: `0.1`

### Output paths

- `paths.checkpoint_dir`: `outputs/checkpoints`
- `paths.plots_dir`: `outputs/plots`
- `paths.results_dir`: `outputs/results`

### Environment variables

No environment variables are required by the current codebase. If you add deployment credentials or a production inference backend later, document them here.

## Usage

### Train the model

```bash
python src/train.py
```

To ignore existing checkpoints and retrain from scratch:

```bash
python src/train.py --fresh
```

What happens during training:

- The dataset is loaded from `config.yaml`.
- Class weights are computed from the training split.
- A DenseNet121 model is created with a custom classifier head.
- Training starts with the backbone frozen.
- At epoch 11, the backbone is unfrozen for fine-tuning.
- The best checkpoint is saved to `outputs/checkpoints/best_model.pth`.
- The most recent state is saved to `outputs/checkpoints/last_model.pth`.
- Training history is written to `outputs/results/history.json`.
- Training curves are written to `outputs/plots/training_curves.png`.

### Evaluate the model

```bash
python src/evaluate.py
```

This script loads the best checkpoint and evaluates the model on the test split. It writes:

- `outputs/results/test_metrics.json`
- `outputs/plots/confusion_matrix.png`
- `outputs/plots/roc_curve.png`
- `outputs/plots/pr_curve.png`
- `outputs/plots/per_class_metrics.png`

### Generate Grad-CAM visualizations

```bash
python src/gradcam.py
```

This creates a Grad-CAM gallery at:

```text
outputs/plots/gradcam_results.png
```

### Run the Streamlit app

```bash
streamlit run app/app.py
```

In the app, users can:

- Upload a chest X-ray image.
- Select from bundled example images.
- Run model inference.
- View the predicted class and confidence.
- Inspect pneumonia/normal probabilities.
- Toggle the Grad-CAM heatmap and adjust its intensity.
- Copy a text report summary.

## API Endpoints (if applicable)

No REST API endpoints are implemented in this repository.

The application is delivered as a Streamlit UI, not as a backend web service. If you later add Flask, FastAPI, or another API layer, list the routes here.

## Screenshots

Add screenshots to support the README and recruiter-facing presentation. Recommended files:

- `docs/screenshots/homepage.png`
- `docs/screenshots/upload-and-prediction.png`
- `docs/screenshots/gradcam-overlay.png`
- `docs/screenshots/training-curves.png`

Suggested placeholders:

| Screenshot | Description |
| --- | --- |
| Project landing view | To Be Added |
| Image upload and prediction result | To Be Added |
| Grad-CAM heatmap overlay | To Be Added |
| Training or evaluation plots | To Be Added |

You can also reuse the generated artifacts already in `outputs/plots/`.

## Model Architecture (if machine learning project)

### Model

The project uses transfer learning with `DenseNet121` as the default architecture. The classifier head is replaced with a custom multilayer head:

- Linear(1024 -> 512)
- BatchNorm1d
- ReLU
- Dropout
- Linear(512 -> 256)
- BatchNorm1d
- ReLU
- Dropout
- Linear(256 -> 2)

An alternative `ResNet50` builder is also implemented in `src/model.py`, but the current configuration uses `densenet121`.

### Dataset

The code expects a chest X-ray dataset organized into three splits:

- `train`
- `val`
- `test`

Each split contains two classes:

- `NORMAL`
- `PNEUMONIA`

The repository is already structured around the Kaggle chest X-ray dataset layout under `data/raw/chest_xray/`.

### Training Process

Training is configured as a two-phase process:

1. Phase 1 trains the classifier head with the backbone frozen.
2. Phase 2 unfreezes the backbone at epoch 11 for full-network fine-tuning.

Other training details:

- Optimizer: Adam
- Learning-rate scheduling: `ReduceLROnPlateau` on validation AUC
- Loss: Cross-entropy with class weights and label smoothing
- Regularization: Dropout and weight decay
- Stability: Gradient clipping with max norm 1.0
- Reproducibility: Random seed fixed to 42

### Evaluation Metrics

The evaluation pipeline reports:

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 score (weighted)
- AUC-ROC
- Sensitivity for `PNEUMONIA`
- Specificity for `NORMAL`
- Per-class precision, recall, and F1

### Reported Results

Based on the saved artifacts in this repository, the best recorded validation checkpoint and test metrics are:

- Best validation AUC: `1.0000`
- Test accuracy: `0.8846`
- Test weighted F1: `0.8843`
- Test AUC-ROC: `0.9436`
- Test sensitivity: `0.9154`
- Test specificity: `0.8333`

Per-class test metrics:

- `NORMAL`: precision `0.8553`, recall `0.8333`, F1 `0.8442`
- `PNEUMONIA`: precision `0.9015`, recall `0.9154`, F1 `0.9084`

Confusion matrix counts:

- TN: `195`
- FP: `39`
- FN: `33`
- TP: `357`

The corresponding training and evaluation outputs are stored under `outputs/results/` and `outputs/plots/`.

## Future Improvements

- Add a formal REST API for programmatic inference.
- Save a reproducible model card with dataset provenance and known limitations.
- Add automated tests for dataset loading, checkpoint loading, and inference.
- Expose a CLI for single-image prediction without the Streamlit UI.
- Add experiment tracking with MLflow or Weights & Biases.
- Provide class-imbalance analysis and calibration plots.
- Add Docker support for consistent local and deployment environments.
- Add model export options such as TorchScript or ONNX.
- Expand explainability with Grad-CAM comparisons across multiple layers.

## Contributing

Contributions are welcome. A good contribution should:

- Follow the existing project structure and coding style.
- Keep changes focused and well documented.
- Update `config.yaml`, the README, or artifact generation code when behavior changes.
- Avoid breaking the existing training and evaluation flow.
- Include validation steps when adding new functionality.

Suggested workflow:

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Validate training, evaluation, or app behavior locally.
5. Open a pull request with a clear description of the change.

## License

To Be Added.

If you intend to publish this project publicly, add a license file such as MIT, Apache 2.0, or BSD 3-Clause, and update this section accordingly.
