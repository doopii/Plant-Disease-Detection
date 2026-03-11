"""
Plant Disease Prediction
Usage: python predict.py <image_path>
   or: python predict.py  (then enter path when prompted)
"""

import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import MobileNet_V3_Large_Weights
from pathlib import Path
from PIL import Image


# Load model
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models_dir = Path('models')
    saved_models = sorted(models_dir.glob('plant_disease_mobilenetv3_best_*.pth'))

    if not saved_models:
        print("No trained model found. Run 03_model_training.ipynb first.")
        sys.exit(1)

    model_path = saved_models[-1]
    print(f"Using model: {model_path.name}")

    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint['classes']
    num_classes = checkpoint['num_classes']

    model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(960, 1280),
        nn.Hardswish(),
        nn.Dropout(p=0.3),
        nn.Linear(1280, num_classes)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, classes, device


# Predict single image
def predict(image_path, model, classes, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        confidence, predicted = torch.max(probs, 0)

    predicted_class = classes[predicted.item()]

    # Top 3 predictions
    top3_probs, top3_idx = torch.topk(probs, 3)

    return predicted_class, confidence.item() * 100, top3_probs, top3_idx, classes


# Main
if __name__ == "__main__":
    model, classes, device = load_model()

    # Get image path from argument or prompt
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("\nEnter image path: ").strip().strip('"')

    image_path = Path(image_path)

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    # Run prediction
    predicted_class, confidence, top3_probs, top3_idx, classes = predict(
        image_path, model, classes, device
    )

    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"Image:      {image_path.name}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print("\nTop 3 Predictions:")
    print("-" * 50)
    for prob, idx in zip(top3_probs, top3_idx):
        print(f"  {classes[idx.item()]:<40} {prob.item()*100:.2f}%")
    print("=" * 50)
