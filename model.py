import torch
from torchvision import transforms
from PIL import Image
import io

# Load the trained model (make sure it's in eval mode)
model = torch.load("model.pth")
model.eval()

# Define a transformation for input image preprocessing (modify based on your model's input)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(image_bytes: bytes):
    # Convert the image bytes into a PIL image
    image = Image.open(io.BytesIO(image_bytes))

    # Apply the transformation (resize, crop, normalize)
    img_tensor = transform(image).unsqueeze(0)

    # Perform prediction (use the model)
    with torch.no_grad():
        output = model(img_tensor)

    # Get the predicted class (for classification models)
    _, predicted_class = torch.max(output, 1)
    return int(predicted_class.item())
