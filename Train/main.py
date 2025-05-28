from train.train import train_and_validate_model
from data.dataloaders import create_dataloaders
from models.mobilenetv3 import mobilenetv3
from models.loss import FocalLoss
import torch
from utils.transforms import train_transform, val_transform
from utils.metrics import calculate_class_weights


batch_size, lr, epochs = 32, 1e-4, 50
train_dir, val_dir = 'dataset/train', 'dataset/valid'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class_columns = ['Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body',
                 'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms']

train_loader, val_loader, train_ds, val_ds = create_dataloaders(
    train_dir, val_dir, train_transform, val_transform, batch_size
)
model = mobilenetv3(num_classes=len(class_columns))
criterion = FocalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


weights = calculate_class_weights(train_ds).to(next(model.parameters()).device)



train_and_validate_model(
    train_loader, val_loader, model, criterion, optimizer,
    num_epochs=epochs,
    output_folder='mobilenet_v3_run',
    class_names=class_columns
)
