import torch
import torch.nn as nn
import gradio as gr
from torchvision import transforms, models
from torchvision.models import MobileNet_V3_Large_Weights
from pathlib import Path
from PIL import Image


# Load model once at startup
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    saved_models = sorted(Path('models').glob('plant_disease_mobilenetv3_best_*.pth'))
    if not saved_models:
        raise FileNotFoundError("No trained model found. Run 03_model_training.ipynb first.")

    checkpoint = torch.load(saved_models[-1], map_location=device)
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
    model.to(device).eval()

    return model, classes, device


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

model, classes, device = load_model()


def predict(image):
    if image is None:
        return {}

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    return {classes[i]: float(probs[i]) for i in range(len(classes))}


# Gradio UI
with gr.Blocks(title="Plant Disease Detection") as app:
    gr.Markdown("# Plant Disease Detection")
    gr.Markdown("Upload a photo of a plant leaf to identify the disease.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Leaf Image")
            predict_btn = gr.Button("Predict", variant="primary")

        with gr.Column():
            label_output = gr.Label(num_top_classes=5, label="Prediction")

    predict_btn.click(fn=predict, inputs=image_input, outputs=label_output)
    image_input.change(fn=predict, inputs=image_input, outputs=label_output)

    gr.Markdown("**Supported plants:** Tomato, Bell Pepper, Chili")

if __name__ == "__main__":
    app.launch()
