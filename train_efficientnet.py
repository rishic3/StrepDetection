import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
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
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# use gpu 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# model definition
model_variant = "efficientnet-b2"
efficientnet = EfficientNetBN(model_variant, spatial_dims=2)

# task definition
SYMPTOM = 'Pus'
NUM_CLASSES = 1
SEED = 1
BATCH_SIZE = 8
num_ftrs = efficientnet._fc.in_features
efficientnet._fc = nn.Linear(num_ftrs, NUM_CLASSES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
efficientnet = efficientnet.to(device)

class StrepDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.at[idx, 'Image_Path']
        image = Image.open(img_name)
        label = int(self.data.at[idx, SYMPTOM])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Transform just for converting the image to tensor
to_tensor_transform = transforms.Compose([
    transforms.Resize((448, 896)),  # Resize to maintain the 2:1 aspect ratio
    transforms.ToTensor()
])

# Create datasets without normalization for computing mean and std
train_dataset_for_mean_std = StrepDataset(csv_file=f'/data/datasets/rishi/symptom_classification/data/combined_train_data_{SYMPTOM}_{SEED}.csv', transform=to_tensor_transform)
data_loader_for_mean_std = DataLoader(train_dataset_for_mean_std, batch_size=64, shuffle=False, num_workers=4)

# Compute mean and std
mean = 0.0
for images, _ in data_loader_for_mean_std:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
mean /= len(data_loader_for_mean_std.dataset)

var = 0.0
for images, _ in data_loader_for_mean_std:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    var += ((images - mean.unsqueeze(1))**2).sum([0,2])
std = torch.sqrt(var / (len(data_loader_for_mean_std.dataset) * 448 * 896))

# Now create the final transform with normalization
transform = transforms.Compose([
    transforms.Resize((448, 896)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Create datasets and dataloaders
#train_dataset = StrepDataset(csv_file=f'/data/datasets/rishi/symptom_classification/data/train_data_{SYMPTOM}_{SEED}.csv', transform=transform)
train_dataset = StrepDataset(csv_file=f'/data/datasets/rishi/symptom_classification/data/combined_train_data_{SYMPTOM}_{SEED}.csv', transform=transform)
test_dataset = StrepDataset(csv_file=f'/data/datasets/rishi/symptom_classification/data/combined_test_data_{SYMPTOM}_{SEED}.csv', transform=transform)

sample_weights = torch.ones(len(train_dataset))
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

"""Training Loop"""

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(efficientnet.parameters(), lr=0.0001)

'''NUM EPOCHS'''
num_epochs = 100

best_combined_metric = 0.0
best_accuracy = 0.0 
best_auc = 0.0
best_model = None
mining = False
mining_start_epoch = 10
mining_freq = 4 if mining else num_epochs * 10

def evaluate(model, test_loader):
    model.eval()
    true_labels = []
    predicted_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze(1)
            predicted_probs.extend(torch.sigmoid(outputs).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    predicted_labels = [1 if prob > 0.5 else 0 for prob in predicted_probs]

    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    auc = roc_auc_score(true_labels, predicted_probs)
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
    
    # Compute PPV, NPV, sensitivity, and specificity
    ppv = tp / (tp + fp) if (tp + fp) != 0 else 0
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    return accuracy, f1, auc, ppv, npv, sensitivity, specificity

def find_hard_negatives(model, data_loader, global_idx_offset):
    model.eval()
    hard_negatives_with_conf = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            predictions = (probabilities > 0.5).astype(int)

            # Identify hard negatives (false positives) and their confidences
            for idx, (pred, prob, label) in enumerate(zip(predictions, probabilities, labels.cpu().numpy())):
                global_idx = idx + global_idx_offset
                if label == 0 and pred == 1:
                    hard_negatives_with_conf.append((global_idx, prob))  # Store global index and confidence

    return hard_negatives_with_conf

global_idx_offset = 0
stagnant_epochs = 0
patience = 20
hard_negatives = []
store_hard_negatives = True

for epoch in range(num_epochs):
    efficientnet.train()
    running_loss = 0.0

    if mining and epoch % mining_freq == 0 and epoch > (mining_start_epoch - 1):
        hard_neg_indices = find_hard_negatives(efficientnet, train_loader, global_idx_offset)
        for idx, conf in hard_neg_indices:
            weight = 2 * conf
            sample_weights[idx] = weight  # Increase the weight of hard negatives
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
        # store some of the hard negatives to visualize later
        if store_hard_negatives:
            hard_negatives = [train_dataset[i][0] for i, _ in hard_neg_indices[:10]]
            store_hard_negatives = False

    global_idx_offset = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        global_idx_offset += len(inputs)
        
        outputs = efficientnet(inputs)

        if outputs.shape[0] != labels.shape[0]:
            raise ValueError(f"Dimension mismatch: outputs {outputs.shape}, labels {labels.shape}")
                
        loss = criterion(outputs.squeeze(1), labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}")
    accuracy, f1, auc, ppv, npv, sensitivity, specificity = evaluate(efficientnet, test_loader)
    print("      Test: Accuracy: {:.4f}, F1 Score: {:.4f}, AUC: {:.4f}, PPV: {:.4f}, NPV: {:.4f}, Sensitivity: {:.4f}, Specificity: {:.4f}".format(accuracy, f1, auc, ppv, npv, sensitivity, specificity))
    
    combined_metric = (accuracy + auc)/2
    if combined_metric > best_combined_metric:
        best_combined_metric = combined_metric
        best_accuracy = accuracy
        best_auc = auc
        best_model = efficientnet.state_dict()
        print("updating model at epoch {}".format(epoch + 1))
        stagnant_epochs = 0
    else:
        stagnant_epochs += 1
        
    if stagnant_epochs == patience:
        break

save_path = f'/data/datasets/rishi/symptom_classification/ckpts/best_{model_variant}_{SYMPTOM}_acc_{round(best_accuracy, 3)}_auc_{round(best_auc, 3)}_seed_{SEED}_mining_{mining}_combined.pth'
if num_epochs > 0:
    torch.save(best_model, save_path)

print('Finished Training')

"""EVAL"""

# visualize hard negatives:
for i, img in enumerate(hard_negatives):
    plt.subplot(2, 5, i+1)
    plt.imshow(img.permute(1, 2, 0).cpu().numpy())
    # save images:
    plt.imsave(f'/data/datasets/rishi/symptom_classification/hard_negatives/{i}.png', img.permute(1, 2, 0).cpu().numpy())

eval = True

'''Saving example images, labels + predictions, saliency maps'''

if eval:
    torch.cuda.empty_cache()
    output_dir = "output_images_combined"
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

    eval_save_path = save_path
    #eval_save_path = '/data/datasets/rishi/symptom_classification/ckpts/best_efficientnet-b3_Pus_acc_0.861_auc_0.922_seed_30_mining_False.pth'
    efficientnet.load_state_dict(torch.load(eval_save_path))
    efficientnet.to(device)

    efficientnet.eval()
    target_layer = [efficientnet._conv_head]
    cam = GradCAM(model=efficientnet, target_layers=target_layer, use_cuda=device.type=='cuda')

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    global_min, global_max = float('inf'), -float('inf')

    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = efficientnet(inputs).squeeze()
        predicted_probs = torch.sigmoid(outputs).detach()

        for i in range(inputs.size(0)):
            image = inputs[i].unsqueeze(0)
            grayscale_cam = cam(input_tensor=image.float())
            grayscale_cam = grayscale_cam[0, :]

            # Update global min and max
            global_min = min(global_min, grayscale_cam.min())
            global_max = max(global_max, grayscale_cam.max())

    # Second Pass: Normalize, Generate CAMs and Save Images
    true_pos_count = 0
    true_neg_count = 0
    false_pos_count = 0
    false_neg_count = 0

    for batch_idx, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = efficientnet(inputs).squeeze()
        predicted_probs = torch.sigmoid(outputs).detach().cpu().numpy()
        predicted_labels = [1 if prob > 0.5 else 0 for prob in predicted_probs]

        for i in range(len(labels)):
            true_label = labels[i].item()
            pred_label = predicted_labels[i]
            image = inputs[i]

            # Generate CAM mask
            grayscale_cam = cam(input_tensor=image.unsqueeze(0).to(device).float())
            grayscale_cam = grayscale_cam[0, :]

            # Normalize the CAM
            grayscale_cam = (grayscale_cam - global_min) / (global_max - global_min)

            # De-normalize and prepare image for saving
            image_for_cam = denormalize(image, mean, std)
            image_for_cam = image_for_cam.permute(1, 2, 0).cpu().numpy()

            # Apply CAM mask
            cam_image = show_cam_on_image(image_for_cam, grayscale_cam, use_rgb=True)

            # Determine the category and save the image and its CAM
            category = categories[pred_label * 2 + true_label]
            image_path = os.path.join(output_dir, category, f"image_{i}-{batch_idx}.png")
            cam_path = os.path.join(output_dir, category, f"cam_{i}-{batch_idx}.png")
            plt.imsave(image_path, image_for_cam)
            plt.imsave(cam_path, cam_image)

            if true_label == pred_label == 1:
                true_pos_count += 1
            elif true_label == pred_label == 0:
                true_neg_count += 1
            elif true_label == 1 and pred_label == 0:
                false_neg_count += 1
            elif true_label == 0 and pred_label == 1:
                false_pos_count += 1

    # Calculate and print metrics
    accuracy = (true_pos_count + true_neg_count) / (true_pos_count + true_neg_count + false_pos_count + false_neg_count)

    print(f"True Positives: {true_pos_count}, True Negatives: {true_neg_count}, False Positives: {false_pos_count}, False Negatives: {false_neg_count}")
    print(f"Accuracy: {(true_pos_count + true_neg_count) / (true_pos_count + true_neg_count + false_pos_count + false_neg_count)}")
    print(f"AUC: {(true_pos_count) / (true_pos_count + false_neg_count)}, PPV: {(true_pos_count) / (true_pos_count + false_pos_count)}, NPV: {(true_neg_count) / (true_neg_count + false_neg_count)}, Sensitivity: {(true_pos_count) / (true_pos_count + false_neg_count)}, Specificity: {(true_neg_count) / (true_neg_count + false_pos_count)}")
    print(f"F1 Score: {(2 * true_pos_count) / (2 * true_pos_count + false_pos_count + false_neg_count)}")