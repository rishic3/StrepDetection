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
from sklearn.metrics import mean_squared_error, r2_score
import copy
from monai.networks.nets import EfficientNetBN
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

efficientnet = EfficientNetBN("efficientnet-b0", spatial_dims=2)

SYMPTOM = 'Tonsil Size'
NUM_CLASSES = 1
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

# Now create the final transform with normalization
transform = transforms.Compose([
    transforms.Resize((448, 896)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Create datasets and dataloaders
train_dataset = StrepDataset(csv_file=f'/data/datasets/rishi/symptom_classification/train_data_{SYMPTOM}.csv', transform=transform)
test_dataset = StrepDataset(csv_file=f'/data/datasets/rishi/symptom_classification/test_data_{SYMPTOM}.csv', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


"""Training Loop"""

criterion = nn.MSELoss()  # Use Mean Squared Error Loss for regression
optimizer = optim.Adam(efficientnet.parameters(), lr=0.0001)
num_epochs = 100

best_model_wts = copy.deepcopy(efficientnet.state_dict())
best_loss = float('inf')
save_path = '/data/datasets/rishi/symptom_classification/best_efficientnet.pth'

# Define your thresholds for converting continuous outputs to discrete classes
thresholds = [0.5, 1.5, 2.5, 3.5, 4.5]

def discretize(predictions, thresholds):
    discrete_preds = []
    for pred in predictions:
        discrete_class = 0
        for threshold in thresholds:
            if pred > threshold:
                discrete_class += 1
        discrete_preds.append(discrete_class)
    return discrete_preds

# Update the evaluate function for regression
def evaluate_regression(model, test_loader, thresholds):
    model.eval()
    total_loss = 0.0
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()  # Ensure labels are float for regression
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            # Convert continuous outputs to discrete classes using thresholds
            preds = discretize(outputs.cpu().numpy(), thresholds)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(preds)

    # Calculate regression metrics, such as Mean Squared Error or R^2 Score
    mse = mean_squared_error(true_labels, predicted_labels)
    r2 = r2_score(true_labels, predicted_labels)

    # Calculate accuracy after discretization
    accuracy = accuracy_score(true_labels, predicted_labels)

    return total_loss, mse, r2, accuracy

for epoch in range(num_epochs):
    efficientnet.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float()  # Ensure labels are float for regression

        optimizer.zero_grad()
        
        outputs = efficientnet(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}")

    # Evaluate the model
    total_loss, mse, r2, accuracy = evaluate_regression(efficientnet, test_loader, thresholds)
    print(f"      Test: Total Loss: {total_loss:.4f}, MSE: {mse:.4f}, R^2: {r2:.4f}, Accuracy: {accuracy:.4f}")

    # Deep copy the model if it has the best loss so far
    if total_loss < best_loss:
        best_loss = total_loss
        best_model_wts = copy.deepcopy(efficientnet.state_dict())

# Load best model weights
efficientnet.load_state_dict(best_model_wts)
torch.save(efficientnet.state_dict(), save_path)

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
