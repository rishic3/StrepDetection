import xml.etree.ElementTree as ET
import pandas as pd
import logging

# Setup logging
logging.basicConfig(filename='/data/datasets/rishi/symptom_classification/data/data_processing.log', level=logging.INFO, format='%(asctime)s: %(message)s')

# Path to your XML file
xml_file = '/data/datasets/rishi/symptom_classification/data/strepannotations_12-1.xml'

# Parse the XML file
logging.info("Parsing XML file...")
tree = ET.parse(xml_file)
root = tree.getroot()

# Extract task (video) details
logging.info("Extracting task details...")
tasks = {}
cumulative_frame_count = 0
frame_offset_per_task = {}

for task in root.find('meta').find('project').find('tasks').findall('task'):
    task_id = task.find('id').text
    task_name = task.find('name').text
    task_size = int(task.find('size').text)  # Total number of frames in the task
    tasks[task_id] = task_name
    frame_offset_per_task[task_id] = cumulative_frame_count
    cumulative_frame_count += task_size
    logging.info(f"Task ID: {task_id}, Video Name: {task_name}, Frame Offset: {frame_offset_per_task[task_id]}")

# Prepare a dictionary to hold our extracted data
data = {}

# Allowed labels
allowed_labels = ["6 Anterior Tonsillar Pillars w Uvula", "8 Right Tonsil plus Pillar"]

# Iterate over each track in the XML
logging.info("Processing tracks...")
for track in root.findall('track'):
    label = track.get('label')
    task_id = track.get('task_id')

    # Skip tracks not matching the allowed labels
    if label not in allowed_labels:
        continue

    video_name = tasks.get(task_id, 'Unknown')
    logging.info(f"Processing track with Task ID: {task_id}, Video Name: {video_name}, Label: {label}")

    # Iterate over each polygon in the track
    for polygon in track.findall('polygon'):
        absolute_frame = int(polygon.get('frame'))
        relative_frame = absolute_frame - frame_offset_per_task[task_id]
        key = (video_name, relative_frame)

        # Initialize data for this key if not present
        if key not in data:
            data[key] = {
                'video_name': video_name,
                'frame': relative_frame,
                'Redness': None, 
                'Pus': None, 
                'Brodsky Tonsil Size': None,
                'Mallampati': None
            }

        # Extract and update attributes
        for attr in polygon.findall('attribute'):
            attr_name = attr.get('name')
            attr_value = attr.text
            if attr_name in data[key]:
                data[key][attr_name] = attr_value

# Convert the dictionary to a DataFrame and export
logging.info("Converting data to DataFrame...")
df = pd.DataFrame(data.values())
csv_file = '/data/datasets/rishi/symptom_classification/data/cvat_labels_12-1.csv'
df.to_csv(csv_file, index=False)
logging.info(f"Data exported to {csv_file}")

print(f'Data exported to {csv_file}')