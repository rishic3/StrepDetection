# StrepDetection

Detection of strep throat directly from cell phone videos.  
Employing intermediate symptom classification combined with rule-based decisions for interpretable results.  
Implementing strategies (hard-negative mining, contrastive learning) to combat limited and imbalanced data.  

## Parsing data from CVAT:  

1. Download data from CVAT
   * `Actions > Export Dataset > Export Format: CVAT for video 1.1`.
   * This will download a folder containing an xml file with the dataset annotations.
3. Parse annotations via `parse_xml.py`  
   * Set the xml file path and run `parse_xml.py`.
   * This will produce a .csv file with the video, frame, and relevant labels.
4. Merge CVAT data with .xlsx data
   * Follow the steps in `data_process.ipynb`.
   * This will merge the annotations from the `.xlsx` training review with the CVAT labels, checking for any overlap.

Authored by Rishi Chandra, rchand18@jhu.edu, as part of the [ARCADE Lab](https://arcade.cs.jhu.edu/) at Johns Hopkins University. 
