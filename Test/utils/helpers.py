import pandas as pd
import numpy as np

def save_predictions_to_excel(image_paths, y_pred, output_path, class_columns):
    y_pred_classes = np.argmax(y_pred, axis=1)
    class_names = [class_columns[i] for i in y_pred_classes]
    df = pd.DataFrame(y_pred, columns=class_columns)
    df.insert(0, 'image_path', image_paths)
    df['predicted_class'] = class_names
    df.to_excel(output_path, index=False)
