import torch
import torch.nn as nn
import os
import pandas as pd
import glob
from torchvision import transforms
from PIL import Image
from monai.networks.nets import EfficientNetBN
from torch.utils.data import Dataset, DataLoader

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to the saved model weights
save_path = '/data/datasets/rishi/symptom_classification/best_efficientnet.pth'

# Load the model
NUM_CLASSES = 1
model = EfficientNetBN("efficientnet-b0", spatial_dims=2)
num_ftrs = model._fc.in_features
model._fc = nn.Linear(num_ftrs, NUM_CLASSES)
model.load_state_dict(torch.load(save_path, map_location=device))
model = model.to(device)
model.eval() 

SYMPTOM = 'Pus'

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
preprocess = transforms.Compose([
    transforms.Resize((448, 896)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Directory containing images to predict on
neg_image_dir = '/data/datasets/rishi/OneDrive_1_11-8-2023/strep_neg_small_tonsilsize_nopus_noredness/'
neg_images = sorted(glob.glob(neg_image_dir + "*.png"))

pos_image_dir = '/data/datasets/rishi/OneDrive_1_11-8-2023/strep_pos_large_tonsilsize_pus_redness/'
pos_images = sorted(glob.glob(pos_image_dir + "*.png"))

# Function to run inference
def run_inference(model, image_path):
    input_image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    with torch.no_grad():
        input_batch = input_batch.to(device)
        output = torch.sigmoid(model(input_batch))
        return output.item()

# Running inference on images and printing the results
for image_path in neg_images:
    prediction = run_inference(model, image_path)
    print(f"Negative Sample: {os.path.basename(image_path)}, {SYMPTOM} probability: {prediction}, {SYMPTOM} classification: {prediction > 0.5}")

print()

for image_path in pos_images:
    prediction = run_inference(model, image_path)
    print(f"Positive Sample: {os.path.basename(image_path)}, {SYMPTOM} probability: {prediction}, {SYMPTOM} classification: {prediction > 0.5}")
