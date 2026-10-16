import json
import os
from megadetector.visualization import visualization_utils as vis_utils
from megadetector.detection import run_detector

def evaluate_terra_incognita(coco_json_path, image_directory):
    # 1. Load the COCO-formatted Camera Traps Data
    print("Loading dataset annotations...")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # 2. Group images by Camera Location ID
    # This is critical for Terra Incognita generalization testing
    locations = {}
    for img in coco_data['images']:
        loc_id = img.get('location')
        if loc_id not in locations:
            locations[loc_id] = []
        locations[loc_id].append(img)
        
    camera_ids = list(locations.keys())
    print(f"Found {len(camera_ids)} unique camera locations in the dataset.")
    
    # 3. Create the Trans-Location (Terra Incognita) Test Split
    # We train on the first 80% of cameras, and test on the remaining 20%
    split_idx = int(len(camera_ids) * 0.8)
    trans_cameras = camera_ids[split_idx:]
    
    trans_test_images = []
    for loc in trans_cameras:
        trans_test_images.extend(locations[loc])
        
    print(f"Terra Incognita Test Set: {len(trans_cameras)} unseen cameras, {len(trans_test_images)} images.")
    
    # 4. Load MegaDetector v5a (Baseline Model)
    print("Downloading and loading MegaDetector v5a...")
    model = run_detector.load_detector('MDV5A')
    
    # 5. Evaluate the model on the unseen Trans-Locations
    # For a real research paper, you would compare your custom model's mAP against these results
    print("Running inference on unseen locations...")
    
    # Testing just the first 10 images for demonstration
    for img_data in trans_test_images[:10]: 
        img_path = os.path.join(image_directory, img_data['file_name'])
        
        if not os.path.exists(img_path):
            continue
            
        # Load the image array 
        image = vis_utils.load_image(img_path)
        
        # Run inference using MegaDetector
        result = model.generate_detections_one_image(image)
        
        # Filter detections above a 50% confidence threshold
        confident_detections = [d for d in result['detections'] if d['conf'] > 0.5]
        
        print(f"File: {img_data['file_name']} | Found: {len(confident_detections)} objects.")

# Example execution:
# evaluate_terra_incognita("caltech_annotations.json", "/content/caltech_images/")