import os
import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

dataset_dir = "asl-dataset/asl_dataset_small"
output_csv = "asl-dataset/landmarks.csv"

with open(output_csv, mode='w', newline='') as f:
    writer = csv.writer(f)
    
    # header: 63 landmark values + label
    header = [f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]] + ["label"]
    writer.writerow(header)
    
    for label in os.listdir(dataset_dir):
        label_path = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_path):
            continue
        
        count = 0
        for img_file in os.listdir(label_path):
            img_path = os.path.join(label_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            
            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0].landmark
                
                # normalize relative to wrist (landmark 0)
                wrist_x, wrist_y, wrist_z = lm[0].x, lm[0].y, lm[0].z
                row = []
                for point in lm:
                    row.extend([
                        point.x - wrist_x,
                        point.y - wrist_y,
                        point.z - wrist_z
                    ])
                row.append(label)
                writer.writerow(row)
                count += 1
        
        print(f"Processed {count} images for: {label}")

print(f"\nDone! Landmarks saved to {output_csv}")
