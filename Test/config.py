import torch

class_columns = [
    'Angioectasia', 'Bleeding', 'Erosion', 'Erythema', 'Foreign Body',
    'Lymphangiectasia', 'Normal', 'Polyp', 'Ulcer', 'Worms'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(class_columns)
