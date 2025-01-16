import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet18_Weights
import torch.optim as optim
import os
import pandas as pd
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from monai.networks.nets import EfficientNetBN
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

efficientnet = EfficientNetBN("efficientnet-b0", spatial_dims=2)

SYMPTOM = 'Tonsil Size'
NUM_CLASSES = 5
num_ftrs = efficientnet._fc.in_features
efficientnet._fc = nn.Linear(num_ftrs, NUM_CLASSES)

print(torch.cuda.is_available())

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
efficientnet = efficientnet.to(device)

class ResizeAndPad:
    def __init__(self, target_height, target_width, desired_size, fill=0, padding_mode='constant'):
        self.target_height = target_height
        self.target_width = target_width
        self.desired_size = desired_size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        # Resize the image to the target size while maintaining aspect ratio.
        img = transforms.Resize((self.target_height, self.target_width))(img)
        
        # Calculate padding to reach the desired size (square).
        pad_height = (self.desired_size - img.size[1]) // 2
        pad_width = (self.desired_size - img.size[0]) // 2

        # Apply padding to make the image square.
        img = F.pad(img, (pad_width, pad_height, pad_width, pad_height), padding_mode=self.padding_mode, fill=self.fill)
        return img

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

# Replace PadToSize with ResizeAndPadToSquare
to_tensor_transform = transforms.Compose([
    ResizeAndPad(448, 896, 896, fill=0),  # Resize and pad to make the image square
    transforms.ToTensor()
])
# Create datasets without normalization for computing mean and std
train_dataset_for_mean_std = StrepDataset(csv_file=f'/data/datasets/rishi/symptom_classification/train_data_{SYMPTOM}.csv', transform=to_tensor_transform)
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

# Now create the final transform with normalization and padding
transform = transforms.Compose([
    ResizeAndPad(448, 896, 896, fill=0),  # Resize and pad to make the image square
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Create datasets and dataloaders
train_dataset = StrepDataset(csv_file=f'/data/datasets/rishi/symptom_classification/train_data_{SYMPTOM}.csv', transform=transform)
test_dataset = StrepDataset(csv_file=f'/data/datasets/rishi/symptom_classification/test_data_{SYMPTOM}.csv', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


"""Training Loop"""

if NUM_CLASSES == 1:
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(efficientnet.parameters(), lr=0.0001)
num_epochs = 100

best_accuracy = 0.0 
best_model = None
save_path = '/data/datasets/rishi/symptom_classification/best_efficientnet.pth'

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

for epoch in range(num_epochs):
    efficientnet.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = efficientnet(inputs)
        if NUM_CLASSES == 1:
            loss = criterion(outputs.squeeze(), labels.float())
        else:
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}")

    if NUM_CLASSES == 1:
        accuracy, f1, auc, ppv, npv, sensitivity, specificity = evaluate(efficientnet, test_loader)
        print("      Test: Accuracy: {:.4f}, F1 Score: {:.4f}, AUC: {:.4f}, PPV: {:.4f}, NPV: {:.4f}, Sensitivity: {:.4f}, Specificity: {:.4f}".format(accuracy, f1, auc, ppv, npv, sensitivity, specificity))
    else: 
        accuracy, f1, auc = evaluate_multiclass(efficientnet, test_loader)
        print(f"      Test: Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = efficientnet.state_dict()

torch.save(best_model, save_path)

print('Finished Training')


"""Displaying example images, labels + predictions"""
if SYMPTOM != 'Tonsil Size':
    # Load the best model
    efficientnet.load_state_dict(torch.load(save_path))
    efficientnet.eval()

    # Fetch and display the requested images
    true_positives, false_positives, true_negatives, false_negatives = [], [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = efficientnet(inputs).squeeze()
            predicted_probs = torch.sigmoid(outputs).cpu().numpy()
            predicted_labels = [1 if prob > 0.5 else 0 for prob in predicted_probs]
            
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = predicted_labels[i]
                if true_label == pred_label == 1 and len(true_positives) < 1:
                    true_positives.append(inputs[i])
                elif true_label == pred_label == 0 and len(true_negatives) < 1:
                    true_negatives.append(inputs[i])
                elif true_label == 1 and pred_label == 0 and len(false_negatives) < 1:
                    false_negatives.append(inputs[i])
                elif true_label == 0 and pred_label == 1 and len(false_positives) < 1:
                    false_positives.append(inputs[i])

    import matplotlib.pyplot as plt

    def denormalize(tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        tensor = tensor.clamp(0, 1)
        return tensor

    output_dir = "output_images"  # change this to your preferred directory
    os.makedirs(output_dir, exist_ok=True)

    samples = [true_negatives[0], false_negatives[0], true_positives[0], false_positives[0]]
    titles = ["True Negative", "False Negative", "True Positive", "False Positive"]

    for sample, title in zip(samples, titles):
        plt.figure(figsize=(5, 5))
        sample = denormalize(sample, mean, std)  # De-normalize the image
        plt.imshow(sample.permute(1, 2, 0).cpu().numpy())
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{title}.png"), bbox_inches='tight')
        plt.close()
