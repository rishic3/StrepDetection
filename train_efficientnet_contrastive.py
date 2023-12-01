import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
import torch.optim as optim
import os
import pandas as pd
import numpy as np
import random
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# task definition
SYMPTOM = 'Pus'
NUM_CLASSES = 1
SEED = 30 
BATCH_SIZE = 8

class CustomEfficientNet(nn.Module):
    def __init__(self, model_variant):
        super(CustomEfficientNet, self).__init__()
        # Initialize the EfficientNet from MONAI
        self.efficientnet = EfficientNetBN(model_variant, spatial_dims=2, in_channels=3, num_classes=1)
        
        # Extract the feature extractor part and the classifier part
        # This might require checking the source code of MONAI's EfficientNetBN to correctly identify these parts
        self.features = nn.Sequential(*list(self.efficientnet.children())[:-1])
        self.classifier = list(self.efficientnet.children())[-1]

    def forward(self, x):
        # Extract features
        x = self.features(x)
        x = torch.flatten(x, 1)
        # Get embeddings
        embeddings = x
        # Class predictions
        preds = self.classifier(embeddings)
        return embeddings, preds
    
model_variant = "efficientnet-b3"
efficientnet = CustomEfficientNet(model_variant)
efficientnet.to(device)

class StrepDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Randomly select another index to form a pair
        idx2 = random.choice(range(len(self.data)))

        # Load first image and label
        img_name = self.data.at[idx, 'Image_Path']
        image1 = Image.open(img_name)
        label1 = int(self.data.at[idx, SYMPTOM])

        # Load second image and label
        img_name2 = self.data.at[idx2, 'Image_Path']
        image2 = Image.open(img_name2)
        label2 = int(self.data.at[idx2, SYMPTOM])

        # Determine if the pair is similar (1) or dissimilar (0)
        pair_label = 1 if label1 == label2 else 0

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return (image1, image2), (label1, pair_label)

# Transform just for converting the image to tensor
to_tensor_transform = transforms.Compose([
    transforms.Resize((448, 896)),  # Resize to maintain the 2:1 aspect ratio
    transforms.ToTensor()
])

# Create datasets without normalization for computing mean and std
train_dataset_for_mean_std = StrepDataset(csv_file=f'/data/datasets/rishi/symptom_classification/train_data_{SYMPTOM}_{SEED}.csv', transform=to_tensor_transform)
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
train_dataset = StrepDataset(csv_file=f'/data/datasets/rishi/symptom_classification/train_data_{SYMPTOM}_{SEED}.csv', transform=transform)
test_dataset = StrepDataset(csv_file=f'/data/datasets/rishi/symptom_classification/test_data_{SYMPTOM}_{SEED}.csv', transform=transform)

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

mining_freq = 5 if mining else num_epochs * 10

def evaluate_multiclass(model, test_loader):
    model.eval()
    true_labels = []
    predicted_labels = []
    probas_list = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probas = torch.nn.functional.softmax(outputs, dim=1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds.cpu().numpy())
            probas_list.extend(probas.cpu().numpy())

    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    auc = roc_auc_score(true_labels, probas_list, multi_class='ovr')
    accuracy = accuracy_score(true_labels, predicted_labels)

    return accuracy, f1, auc

def evaluate(model, test_loader):
    model.eval()
    true_labels = []
    predicted_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
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

def contrastive_loss(embeddings1, embeddings2, label, margin=1.0):
    # Normalize embeddings
    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    embeddings2 = F.normalize(embeddings2, p=2, dim=1)

    # Cosine similarity
    cosine_similarity = F.cosine_similarity(embeddings1, embeddings2)

    # Calculate loss
    loss_similar = (1 - label) * (1 - cosine_similarity).pow(2)  # For similar pairs
    loss_dissimilar = label * F.relu(cosine_similarity - margin).pow(2)  # For dissimilar pairs

    # Average loss
    loss = torch.mean(loss_similar + loss_dissimilar)

    return loss

def find_hard_negatives(model, data_loader):
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
                if label == 0 and pred == 1:
                    hard_negatives_with_conf.append((idx, prob))  # Store index and confidence

    return hard_negatives_with_conf

global_idx_offset = 0
stagnant_epochs = 0
patience = 20

for epoch in range(num_epochs):
    efficientnet.train()
    running_loss = 0.0

    if mining and epoch % mining_freq == 0 and epoch > (mining_start_epoch - 1):
        hard_neg_indices = find_hard_negatives(efficientnet, train_loader)
        for idx, conf in hard_neg_indices:
            global_idx = idx + global_idx_offset
            weight = 2 * conf
            sample_weights[global_idx] = weight # Increase the weight of hard negatives
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)

    global_idx_offset += len(train_loader.dataset)

    for (inputs1, inputs2), (labels, pair_labels) in train_loader:
        inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
        labels, pair_labels = labels.to(device), pair_labels.to(device)

        optimizer.zero_grad()

        # Get embeddings and class predictions
        embeddings1, preds1 = efficientnet(inputs1)
        embeddings2, preds2 = efficientnet(inputs2)

        # Compute classification loss (assuming binary classification)
        loss1 = criterion(preds1.squeeze(), labels.float())
        loss2 = criterion(preds2.squeeze(), labels.float())

        # Compute contrastive loss
        cont_loss = contrastive_loss(embeddings1, embeddings2, pair_labels)

        # Combine losses
        total_loss = loss1 + loss2 + cont_loss

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item() * inputs1.size(0)

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
        stagnant_epochs = 0
    else:
        stagnant_epochs += 1
        if stagnant_epochs == patience:
            break

save_path = f'/data/datasets/rishi/symptom_classification/ckpts/best_{model_variant}_{SYMPTOM}_acc_{round(best_accuracy, 3)}_auc_{round(best_auc, 3)}_seed_{SEED}_mining_{mining}.pth'
if num_epochs > 0:
    torch.save(best_model, save_path)

print('Finished Training')

"""EVAL"""

eval = False

'''Saving example images, labels + predictions, saliency maps'''

if eval:
    torch.cuda.empty_cache()
    output_dir = "data/output_images"
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
    #eval_save_path = '/data/datasets/rishi/symptom_classification/best_efficientnet_Pus_acc_0.832_auc_0.906_seed_30_mining_False.pth'
    efficientnet.load_state_dict(torch.load(eval_save_path))
    efficientnet.to(device)

    efficientnet.eval()
    target_layer = [efficientnet._conv_head]
    cam = GradCAM(model=efficientnet, target_layers=target_layer, use_cuda=device.type=='cuda')

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

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
            
            image_path = os.path.join(output_dir, category, f"image_{i}-{batch_idx}.png")
            cam_path = os.path.join(output_dir, category, f"cam_{i}-{batch_idx}.png")
            plt.imsave(image_path, image_for_cam)
            plt.imsave(cam_path, cam_image)

    print(f"True Positives: {true_pos_count}, True Negatives: {true_neg_count}, False Positives: {false_pos_count}, False Negatives: {false_neg_count}")
    print(f"Accuracy: {(true_pos_count + true_neg_count) / (true_pos_count + true_neg_count + false_pos_count + false_neg_count)}")