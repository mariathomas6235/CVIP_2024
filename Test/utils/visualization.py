import numpy as np
import cv2

def overlay_heatmap_on_image(image_np, heatmap):
    heatmap = np.mean(heatmap, axis=2) if heatmap.ndim == 3 else heatmap
    heatmap = np.clip((heatmap - heatmap.min()) / (heatmap.ptp() + 1e-8), 0, 1)
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    return cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

