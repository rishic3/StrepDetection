import json
import csv

# Load COCO JSON annotations
with open('/data/datasets/rishi/symptom_classification/instances_default.json', 'r') as json_file:
    data = json.load(json_file)

# Define the label IDs you want to extract
label_ids_to_extract = {0, 1, 2, 3}

# Create a dictionary to map category IDs to category names
category_id_to_name = {category['id']: category['name'] for category in data['categories']}


do_print = True
# CSV file to save the labels
with open('/data/datasets/rishi/symptom_classification/redness_labels.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Image_ID', 'Label'])  # Writing the headers

    # Extract labels from the annotations
    for annotation in data['annotations']:
        if do_print:
            print(annotation.keys())
            do_print = False
            print("id:", annotation['id'])
            print("image id:", annotation['image_id'])
            print("category id:", annotation['category_id'])
            print("attributes:", annotation['attributes'])
        image_id = annotation['image_id']
        label_id = annotation['category_id']
        
        if label_id in label_ids_to_extract:
            label_name = category_id_to_name[label_id]
            writer.writerow([image_id, label_name])