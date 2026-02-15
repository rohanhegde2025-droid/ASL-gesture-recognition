import os
import shutil
import random

source_dir = "asl-dataset/asl_alphabet_train"        
dest_dir = "asl-dataset/asl_dataset_small"    
samples_per_class = 700

for letter_folder in os.listdir(source_dir):
    src_folder = os.path.join(source_dir, letter_folder)
    
    if not os.path.isdir(src_folder):
        continue
    
    all_images = os.listdir(src_folder)
    
    sampled = random.sample(all_images, min(samples_per_class, len(all_images)))
    
    dest_folder = os.path.join(dest_dir, letter_folder)
    os.makedirs(dest_folder, exist_ok=True)
    
    for img in sampled:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(dest_folder, img)
        )
    
    print(f"Copied {len(sampled)} images for: {letter_folder}")

print("Done!", dest_dir)