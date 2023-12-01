import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

file_path = "/data/datasets/rishi/symptom_classification/data/training_review_tc_3.14.23.xlsx"
df = pd.read_excel(file_path)

SYMPTOM = 'Pus'
SEED = 30

# Remove missing entries (for now, just tonsil size)
df = df[df[SYMPTOM] != 99]

# Extract patient IDs and frame names
df['Patient_ID'] = df['Frame Name'].str.extract(r'^(.*?)-frame_')
df['Frame'] = df['Frame Name'].str.extract(r'(frame_\d+)')

print("Number of unique patients:", len(df['Patient_ID'].unique()))

# Compute the image paths based on patient IDs and frame names
df['Image_Path'] = '/data/datasets/rishi/cropped_frames/' + df['Patient_ID'] + '/' + df['Frame'] + '.jpg'
df = df[df['Image_Path'].apply(os.path.exists)]

print("Total number of frames:", df.shape[0])
positive = df[df[SYMPTOM] == 1].shape[0]
negative = df[df[SYMPTOM] == 0].shape[0]

# Print the number and proportion of positive and negative frames for symptom
print(f"Number of positive frames for {SYMPTOM}:", positive)
print(f"Proportion of positive frames for {SYMPTOM}:", positive/df.shape[0])
print(f"Number of negative frames for {SYMPTOM}:", negative)
print(f"Proportion of negative frames for {SYMPTOM}:", negative/df.shape[0])

patient_ids = df['Patient_ID'].unique()

# Split data into train and test based on patient IDs
train_patient_ids, test_patient_ids = train_test_split(patient_ids, test_size=0.3, random_state=SEED)

# Filter dataframe rows based on train/test patient IDs
train_df = df[df['Patient_ID'].isin(train_patient_ids)]
test_df = df[df['Patient_ID'].isin(test_patient_ids)]

# Select desired columns
columns_to_keep = ['Patient_ID', 'Image_Path', 'Mallampati', 'In Focus', 'Use to Train', 'Redness', 'Pus', 'Tonsil Size', 'Strep Positive']
train_df = train_df[columns_to_keep]
test_df = test_df[columns_to_keep]

# Save the train and test dataframe as CSV
train_csv_path = f'/data/datasets/rishi/data/symptom_classification/train_data_{SYMPTOM}_{SEED}.csv'
test_csv_path = f'/data/datasets/rishi/data/symptom_classification/test_data_{SYMPTOM}_{SEED}.csv'

image_sizes = []

for index, row in train_df.iterrows():
    with Image.open(row['Image_Path']) as img:
        image_sizes.append(img.size)

max_width = max([size[0] for size in image_sizes])
max_height = max([size[1] for size in image_sizes])
average_width = sum([size[0] for size in image_sizes]) / len(image_sizes)
average_height = sum([size[1] for size in image_sizes]) / len(image_sizes)

print("average width, height:", average_width, average_height)
print("max width, height:", max_width, max_height)

train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)