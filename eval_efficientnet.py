import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
import torch.optim as optim
import os
import pandas as pd
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from monai.networks.nets import EfficientNetBN
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

"""EVAL"""
'''Saving example images, labels + predictions, saliency maps'''

output_dir = "output_images"
categories = ["True_Negatives", "False_Negatives", "True_Positives", "False_Positives"]
for category in categories:
    os.makedirs(os.path.join(output_dir, category), exist_ok=True)

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    tensor = tensor.clamp(0, 1)
    return tensor

'''Define model and gradcam'''
efficientnet = EfficientNetBN(model_variant, spatial_dims=2)
num_ftrs = efficientnet._fc.in_features
efficientnet._fc = nn.Linear(num_ftrs, NUM_CLASSES)

#eval_save_path = save_path
eval_save_path = '/data/datasets/rishi/symptom_classification/best_efficientnet_Pus_acc_0.832_auc_0.906_seed_30_mining_False.pth'
efficientnet.load_state_dict(torch.load(eval_save_path))

target_layer = efficientnet._conv_head
cam = GradCAM(model=efficientnet, target_layers=target_layer, use_cuda=device.type=='cuda')

with torch.no_grad():
    true_pos_count = 0
    true_neg_count = 0
    false_pos_count = 0
    false_neg_count = 0
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = efficientnet(inputs).squeeze()
        predicted_probs = torch.sigmoid(outputs).cpu().numpy()
        predicted_labels = [1 if prob > 0.5 else 0 for prob in predicted_probs]

        for i in range(len(labels)):
            true_label = labels[i].item()
            pred_label = predicted_labels[i]
            image = inputs[i]

            # Generate CAM mask
            grayscale_cam = cam(input_tensor=image.unsqueeze(0), target_category=pred_label)
            grayscale_cam = grayscale_cam[0, :]

            # De-normalize and prepare image for saving
            image_for_cam = denormalize(image, mean, std)  # Replace 'mean' and 'std' with your values
            image_for_cam = image_for_cam.permute(1, 2, 0).cpu().numpy()

            # Apply CAM mask
            cam_image = show_cam_on_image(image_for_cam, grayscale_cam, use_rgb=True)

            # Determine the category and save the image and its CAM
            if true_label == pred_label == 1:
                category = "True_Positives"
                true_pos_count += 1
            elif true_label == pred_label == 0:
                category = "True_Negatives"
                true_neg_count += 1
            elif true_label == 1 and pred_label == 0:
                category = "False_Negatives"
                false_neg_count += 1
            elif true_label == 0 and pred_label == 1:
                category = "False_Positives"
                false_pos_count += 1
            
            image_path = os.path.join(output_dir, category, f"image_{i}.png")
            cam_path = os.path.join(output_dir, category, f"cam_{i}.png")
            plt.imsave(image_path, image_for_cam)
            plt.imsave(cam_path, cam_image)