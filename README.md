# 🌿 Plant Disease Detection

A deep learning-based plant disease detection system using MobileNetV3 to identify diseases in chili, pepper, and tomato plants.

## 📋 Overview

This project uses transfer learning with MobileNetV3 to classify plant diseases across three types of plants:
- **Chili**: Healthy, Leaf curl, Leaf spot, Whitefly, Yellowish
- **Pepper (Bell)**: Healthy, Bacterial spot
- **Tomato**: Healthy, Early blight, Late blight, Bacterial spot, Leaf Mold

## 🚀 Features

- Transfer learning using MobileNetV3 Large architecture
- Multi-class classification (12 disease categories)
- Data augmentation for improved model generalization
- Model checkpointing and backup system
- GPU acceleration support

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/doopii/Leafy-Disease-Detection.git
cd Leafy-Disease-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🗂️ Project Structure

```
├── main.ipynb                  # Main training notebook
├── models/                     # Saved model checkpoints
│   └── plant_disease_mobilenetv3.pth
├── data/                       # Dataset directory
│   ├── train/                  # Training images
│   └── valid/                  # Validation images
└── requirements.txt            # Python dependencies
```

## 💻 Usage

1. Open and run the Jupyter notebook:
```bash
jupyter notebook main.ipynb
```

2. The notebook includes:
   - Dataset download and preparation
   - Data augmentation and preprocessing
   - Model training with MobileNetV3
   - Model evaluation and testing
   - Inference on new images

## 🎯 Model Architecture

- **Base Model**: MobileNetV3 Large (pretrained on ImageNet)
- **Input Size**: 224x224 pixels
- **Output Classes**: 12 disease categories
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss

## 📊 Dataset

The model is trained on combined datasets from Kaggle:
- New Plant Diseases Dataset (Tomato & Pepper)
- Chili Plant Disease Dataset
- Additional Chili Plant Diseases Dataset

## 🔧 Training

The training process includes:
- Data augmentation (rotation, flipping, color jittering)
- Normalization using ImageNet statistics
- Learning rate scheduling
- Automatic model checkpointing

## 📈 Results

Model checkpoints are saved in the `models/` directory with automatic backup functionality.

## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests.

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

Created by [doopii](https://github.com/doopii)

## 🙏 Acknowledgments

- Datasets from Kaggle contributors
- PyTorch and torchvision teams
- MobileNetV3 architecture by Google Research
