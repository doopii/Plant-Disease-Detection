# Plant Disease Detection

A deep learning-based plant disease detection system using MobileNetV3 to identify diseases in chili, pepper, and tomato plants.

## Overview

This project uses transfer learning with MobileNetV3 to classify plant diseases across three types of plants:
- **Chili**: Healthy, Leaf curl, Leaf spot, Whitefly, Yellowish
- **Pepper (Bell)**: Healthy, Bacterial spot
- **Tomato**: Healthy, Early blight, Late blight, Bacterial spot, Leaf Mold

## Features

- Transfer learning using MobileNetV3 Large architecture
- Multi-class classification (12 disease categories)
- ~98% validation accuracy
- Data augmentation for improved model generalization
- Model checkpointing and backup system
- GPU acceleration support
- ONNX and TorchScript exports for mobile deployment

## Installation

1. Clone the repository:
```bash
git clone https://github.com/doopii/Plant-Disease-Detection.git
cd Plant-Disease-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── notebooks/
│   ├── 01_data_preparation.ipynb       # Dataset download and setup
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_model_training.ipynb         # Model training
│   ├── 04_model_evaluation.ipynb       # Evaluation and metrics
│   └── 05_model_conversion.ipynb       # Export for mobile deployment
├── models/
│   ├── plant_disease_mobilenetv3.pth           # PyTorch checkpoint
│   ├── plant_disease_mobilenetv3.onnx          # ONNX export
│   ├── plant_disease_mobilenetv3.torchscript.pt
│   ├── plant_disease_mobilenetv3_quantized.torchscript.pt
│   └── model_metadata.json                     # Class labels and input specs
├── mobile/
│   ├── plant_disease_mobilenetv3.onnx          # ONNX model for Flutter/Android
│   └── model_metadata.json
├── app.py                  # Gradio web demo
├── predict.py              # CLI inference script
├── data/                   # Dataset directory
│   ├── train/
│   ├── val/
│   └── test/
└── requirements.txt
```

## Usage

Run predictions on an image:
```bash
python predict.py path/to/image.jpg
```

Launch the Gradio web demo:
```bash
python app.py
```

To retrain or explore the model, run the notebooks in order (01 through 05).

## Model Architecture

- **Base Model**: MobileNetV3 Large (pretrained on ImageNet)
- **Classifier**: Linear(960 → 1280) → Hardswish → Dropout(0.3) → Linear(1280 → 12)
- **Input Size**: 224x224 pixels
- **Output Classes**: 12 disease categories
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss

## Mobile Deployment (Flutter/Android)

The `mobile/` folder contains everything needed for on-device inference:
- `plant_disease_mobilenetv3.onnx` — use with the [`onnxruntime`](https://pub.dev/packages/onnxruntime) Flutter package
- `model_metadata.json` — class labels and normalization parameters (mean/std)

Input preprocessing: resize to 224x224, normalize with mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.

## Dataset

The model is trained on combined datasets from Kaggle:
- New Plant Diseases Dataset (Tomato & Pepper)
- Chili Plant Disease Dataset
- Additional Chili Plant Diseases Dataset

## Training

The training process includes:
- Data augmentation (rotation, flipping, color jittering)
- Normalization using ImageNet statistics
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping (patience=5)
- Automatic model checkpointing

## Results

| Format | Size | Inference Speed |
|---|---|---|
| PyTorch | 16.30 MB | baseline |
| TorchScript | 16.72 MB | 1.23x faster |
| Quantized TorchScript | 13.20 MB | 1.32x faster |
| ONNX Runtime | 16.08 MB | 4.03x faster |

## License

This project is open source and available under the MIT License.

## Author

Created by [doopii](https://github.com/doopii)

## Acknowledgments

- Datasets from Kaggle contributors
- PyTorch and torchvision teams
- MobileNetV3 architecture by Google Research
