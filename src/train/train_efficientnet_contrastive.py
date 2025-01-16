import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
import torch.optim as optim
import os
import pandas as pd
import numpy as np
import random
import torch.nn.functional as F
from torchvision import transforms
#from torchvision.transforms import functional as F
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

class CustomEfficientNet(nn.Module):
    def __init__(self, model_variant):
        super(CustomEfficientNet, self).__init__()
        # Initialize the EfficientNet from MONAI
        self.efficientnet = EfficientNetBN(model_variant, spatial_dims=2, in_channels=3, num_classes=1)

    def forward(self, x):
        # Pass the input through EfficientNet up to the _avg_pooling layer
        x = self.efficientnet._conv_stem(x)
        x = self.efficientnet._bn0(x)
        x = self.efficientnet._swish(x)

        # Pass through each MBConvBlock
        for block in self.efficientnet._blocks:
            x = block(x)

        x = self.efficientnet._conv_head(x)
        x = self.efficientnet._bn1(x)
        x = self.efficientnet._swish(x)

        # The output of this layer can be used as embeddings
        embeddings = self.efficientnet._avg_pooling(x)

        # Flatten the embeddings for the classifier
        embeddings_flattened = torch.flatten(embeddings, 1)

        # Pass through the classifier (_fc layer)
        preds = self.efficientnet._fc(embeddings_flattened)

        return embeddings, preds

model_variant = "efficientnet-b3"
efficientnet = CustomEfficientNet(model_variant)
efficientnet.to(device)

class ContrastiveStrepDataset(Dataset):
    def __init__(self, base_dataset):
        """
        Initialize the wrapper with the base dataset (StrepDataset instance).
        """
        self.base_dataset = base_dataset

    def __len__(self):
        """
        The length of the dataset is the same as the base dataset.
        """
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        Returns a pair of images, their individual labels, 
        and a label indicating whether they are similar (1) or dissimilar (0).
        """
        # Get the first image and label
        img1, label1 = self.base_dataset[idx]

        # Randomly select another index for the second image
        idx2 = idx
        while idx2 == idx:
            idx2 = random.choice(range(len(self.base_dataset)))

        img2, label2 = self.base_dataset[idx2]

        # Determine if the pair is similar (1) or dissimilar (0)
        pair_label = 1 if label1 == label2 else 0

        return (img1, img2), (label1, label2), pair_label

# Transform just for converting the image to tensor
to_tensor_transform = transforms.Compose([
    transforms.Resize((448, 896)),  # Resize to maintain the 2:1 aspect ratio
    transforms.ToTensor()
])

# Create datasets without normalization for computing mean and std
train_dataset_for_mean_std = StrepDataset(csv_file=f'/data/datasets/rishi/symptom_classification/data/combined_train_data_Pus.csv', transform=to_tensor_transform)
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
base_train_dataset = StrepDataset(csv_file=f'/data/datasets/rishi/symptom_classification/data/combined_train_data_Pus.csv', transform=transform)
base_test_dataset = StrepDataset(csv_file=f'/data/datasets/rishi/symptom_classification/data/test_data_{SYMPTOM}_{SEED}.csv', transform=transform)

train_dataset = ContrastiveStrepDataset(base_train_dataset)
test_dataset = ContrastiveStrepDataset(base_test_dataset)

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
MINING = False
mining_start_epoch = 5

mining_freq = 5 if MINING else num_epochs * 10

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
        for data in test_loader:
            ((inputs1, inputs2), (labels1, labels2), _) = data

            # Process both images in the pair and concatenate for a batch-wise operation
            inputs_combined = torch.cat([inputs1, inputs2], dim=0).to(device)
            labels_combined = torch.cat([labels1, labels2], dim=0).to(device)

            # Get predictions for the combined batch
            _, outputs_combined = model(inputs_combined)
            outputs_combined = outputs_combined.squeeze()

            predicted_probs.extend(torch.sigmoid(outputs_combined).cpu().numpy())
            true_labels.extend(labels_combined.cpu().numpy())

    predicted_labels = [1 if prob > 0.5 else 0 for prob in predicted_probs]

    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    auc = roc_auc_score(true_labels, predicted_probs)
    
    # Compute confusion matrix and other metrics
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
    ppv = tp / (tp + fp) if (tp + fp) != 0 else 0  # Positive Predictive Value
    npv = tn / (tn + fn) if (tn + fn) != 0 else 0  # Negative Predictive Value
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0  # Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0  # Specificity

    return accuracy, f1, auc, ppv, npv, sensitivity, specificity

def contrastive_loss(embeddings1, embeddings2, label, margin=1.0):
    # Manually normalize embeddings
    norm1 = torch.norm(embeddings1, p=2, dim=1, keepdim=True)
    norm2 = torch.norm(embeddings2, p=2, dim=1, keepdim=True)
    embeddings1 = embeddings1 / norm1
    embeddings2 = embeddings2 / norm2

    # Cosine similarity
    cosine_similarity = torch.sum(embeddings1 * embeddings2, dim=1)

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
        for ((inputs1, inputs2), _, _) in data_loader:
            inputs1, inputs2 = inputs1.to(device), inputs2.to(device)

            # Forward pass for each image in the pair and combine predictions
            _, preds1 = model(inputs1)
            _, preds2 = model(inputs2)
            preds_combined = torch.cat([preds1, preds2], dim=0).squeeze()

            probabilities = torch.sigmoid(preds_combined).cpu().numpy()
            predictions = (probabilities > 0.5).astype(int)

            # The labels for each pair are combined for hard negative mining
            for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
                if idx % 2 == 0:  # Only consider one image from each pair to avoid duplication
                    if pred == 1:  # Hard negative identified (assuming label 0 is negative)
                        hard_negatives_with_conf.append((idx // 2, prob))  # Store pair index and confidence

    return hard_negatives_with_conf

global_idx_offset = 0
stagnant_epochs = 0
patience = 40

for epoch in range(num_epochs):
    efficientnet.train()
    running_loss = 0.0

    if MINING and epoch % mining_freq == 0 and epoch > (mining_start_epoch - 1):
        hard_neg_indices = find_hard_negatives(efficientnet, train_loader)
        for idx, conf in hard_neg_indices:
            pair_idx = idx  # Index of the pair
            weight = 2 * conf
            sample_weights[pair_idx] = weight  # Increase the weight of hard negative pairs

        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)

    global_idx_offset += len(train_loader.dataset) // 2  # Divide by 2 as each item is a pair

    for data in train_loader:
        ((inputs1, inputs2), (labels1, labels2), pair_labels) = data
        inputs1, inputs2 = inputs1.to(device), inputs2.to(device)
        labels1, labels2 = labels1.to(device), labels2.to(device)
        pair_labels = pair_labels.to(device)

        optimizer.zero_grad()

        # Get embeddings and class predictions
        embeddings1, preds1 = efficientnet(inputs1)
        embeddings2, preds2 = efficientnet(inputs2)

        # Compute binary classification loss for each image
        loss_class1 = criterion(preds1.squeeze(), labels1.float())
        loss_class2 = criterion(preds2.squeeze(), labels2.float())
        loss_class = (loss_class1 + loss_class2) / 2

        # Compute contrastive loss
        cont_loss = contrastive_loss(embeddings1, embeddings2, pair_labels.float())

        # Combine losses
        total_loss = loss_class + cont_loss

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

save_path = f'/data/datasets/rishi/symptom_classification/ckpts/best_{model_variant}_{SYMPTOM}_acc_{round(best_accuracy, 3)}_auc_{round(best_auc, 3)}_seed_{SEED}_mining_{MINING}_contrastive.pth'
if num_epochs > 0:
    torch.save(best_model, save_path)

print('Finished Training')

"""EVAL"""

eval = True

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
    #eval_save_path = '/data/datasets/rishi/symptom_classification/ckpts/best_efficientnet_Pus_acc_0.806_auc_0.842_mining_True.pth'
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
    print(f"Accuracy: {accuracy}")