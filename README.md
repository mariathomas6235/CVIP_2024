# CVIP_2024
Capsule Vision 2024 Challenge: Multi-Class Abnormality Classification for Video Capsule Endoscopy (VCE), organized by the Research Center for Medical Image Analysis and Artificial Intelligence (MIAAI), Department of Medicine, Danube Private University, Krems, Austria, and the Medical Imaging and Signal Analysis Hub (MISAHUB), in collaboration with the 9th International Conference on Computer Vision & Image Processing (CVIP 2024), being organized by the Indian Institute of Information Technology, Design and Manufacturing (IIITDM) Kancheepuram, Chennai, India.

The objective of the challenge was to develop AI/ML-based models to classify abnormalities in VCE video frames.

In this challenge, deep learning-based models (ResNet18, DenseNet121, MobileNetV3, etc.) were explored for the classification of VCE images. We incorporated focal loss to handle class imbalance and self-attention mechanisms to capture contextual details. The MobileNetV3 model with self-attention and focal loss achieved average sensitivity, average F1-score, average precision, and balanced accuracy of 0.82, 0.84, 0.89, and 0.83, respectively, on the provided validation data.

Our team secured 13th place in the Capsule Vision 2024 Challenge, which saw participation from 150 teams worldwide. Out of these, 35 teams submitted entries, and 27 were selected for final evaluation after review. Best performing model, MobileNetV3 with attention, has a Mean AUC of 0.69, combined metric of 0.47, average specificity of 0.91, average F1-score of 0.19, average precision of 0.25, and balanced accuracy of 0.24 on the testing dataset.


