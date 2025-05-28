from config import *
from dataset import CustomTestDataset
from model import MobileNetV3WithAttention
from utils.gradcam import generate_gradcam
from utils.visualization import overlay_heatmap_on_image
from utils.helpers import save_predictions_to_excel
from utils.metrics import generate_metrics_report

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

model = MobileNetV3WithAttention(num_classes=num_classes)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device).eval()


test_folder = 'dataset'
dataset = CustomTestDataset(test_folder, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=False)


os.makedirs('test_results/gradcam', exist_ok=True)


all_preds = []
all_image_names = []
all_pred_labels = []

with torch.no_grad():
    for images, image_names in loader:
        images = images.to(device)

        
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_preds.append(probs)
        all_pred_labels.extend(preds)
        all_image_names.extend(image_names)

        
        for i in range(images.size(0)):
            orig_path = os.path.join(test_folder, image_names[i])
            orig_img = Image.open(orig_path).convert('RGB').resize((256, 256))
            
            heatmap = generate_gradcam(model, images[i].unsqueeze(0), class_idx=preds[i])
            overlay = overlay_heatmap_on_image(orig_img, heatmap)
            
            save_path = os.path.join('test_results/gradcam', f'{os.path.splitext(image_names[i])[0]}_gradcam.jpg')
            overlay.save(save_path)

all_preds = np.concatenate(all_preds, axis=0)
save_predictions_to_excel(all_image_names, all_preds, 'test_results/predictions.xlsx', class_columns)


