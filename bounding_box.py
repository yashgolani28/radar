from ultralytics import YOLO
import cv2
import os

try:
    model = YOLO("human_vehicel_weight.pt")
    print("[INFO] Custom model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None

def annotate_speeding_object(image_path, radar_distance, label=None, save_dir="snapshots", min_confidence=0.2):
    print(f"[DEBUG] Annotating snapshot: {image_path}")
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"[ERROR] Failed to read image: {image_path}")
        return None
    
    results = model(img)[0]  # This is where YOLO processes the image.
    h, w = img.shape[:2]
    best_box = None
    min_distance_delta = float("inf")

    for box in results.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf[0])

        if conf < min_confidence:
            print(f"[DEBUG] Skipping {cls_id} with low confidence {conf}")
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_width = x2 - x1
        estimated_distance = max(0.1, 5000 / (box_width + 1))
        distance_delta = abs(estimated_distance - radar_distance)

        if distance_delta < min_distance_delta:
            min_distance_delta = distance_delta
            best_box = (x1, y1, x2, y2, model.names[cls_id], conf)

    if best_box:
        x1, y1, x2, y2, cls_name, conf = best_box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = f"{cls_name.upper()} ({conf:.2f})"
        if label:
            text += f" | {label}"
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        print(f"[DEBUG] Bounding box on {cls_name}")
    else:
        print("[DEBUG] No bounding box detected")

    output_path = os.path.join(save_dir, "annotated_" + os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    return output_path

