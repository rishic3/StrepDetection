import xml.etree.ElementTree as ET
import pandas as pd

# Path to your XML file
xml_file = '/data/datasets/rishi/symptom_classification/data/annotations_full.xml'

# Parse the XML file
tree = ET.parse(xml_file)
root = tree.getroot()

# Extract task (video) details
tasks = {}
for task in root.find('meta').find('project').find('tasks').findall('task'):
    task_id = task.find('id').text
    task_name = task.find('name').text
    tasks[task_id] = task_name

# Prepare a list to hold our extracted data
data = []

# Iterate over each track in the XML
for track in root.findall('track'):
    label = track.get('label')
    task_id = track.get('task_id')
    video_name = tasks.get(task_id, 'Unknown')

    if label == "8 Right Tonsil plus Pillar" or label == "6 Anterior Tonsillar Pillars w Uvula":
        # Iterate over each polygon in the track
        for polygon in track.findall('polygon'):
            frame = polygon.get('frame')
            redness = pus = tonsil_size = mallampati = None

            # Extract attributes based on the label
            if label == "8 Right Tonsil plus Pillar":
                redness = polygon.find(".//attribute[@name='Redness']").text
                pus = polygon.find(".//attribute[@name='Pus']").text
                tonsil_size = polygon.find(".//attribute[@name='Brodsky Tonsil Size']").text
            elif label == "6 Anterior Tonsillar Pillars w Uvula":
                mallampati_attr = polygon.find(".//attribute[@name='Mallampati']")
                mallampati = mallampati_attr.text if mallampati_attr is not None else 'Unknown'

            data.append({
                'video_name': video_name,
                'frame': frame,
                'label': label,
                'Redness': redness,
                'Pus': pus,
                'Brodsky Tonsil Size': tonsil_size,
                'Mallampati': mallampati
            })

# Convert the list to a DataFrame
df = pd.DataFrame(data)

# Export the DataFrame to a CSV file
csv_file = '/data/datasets/rishi/symptom_classification/data/cvat_symptom_labels.csv'
df.to_csv(csv_file, index=False)

print(f'Data exported to {csv_file}')
