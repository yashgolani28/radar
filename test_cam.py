from ultralytics import YOLO
import cv2
from bounding_box import annotate_speeding_object
radar_distance= 10
model = YOLO("human_vehicel_weight.pt")
img_path = "C:\ESSI\Projects\Final-radar-detection-system\snapshots\speeding_20250619_155522_352.jpg" 
annotate_speeding_object(img_path, radar_distance, label=None, save_dir="snapshots", min_confidence=0.2)
'''img = cv2.imread(img_path)

results = model(img)[0]

for box in results.boxes:
    cls_id = int(box.cls)
    conf = float(box.conf[0])
    cls_name = model.names[cls_id]
    print(f"Detected {cls_name} with confidence {conf:.2f}")'''
