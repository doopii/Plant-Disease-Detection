"""
Complete Dataset Setup Script
Downloads Kaggle datasets and merges with Roboflow data
"""

import os
import shutil
import kagglehub

print("=" * 70)
print("COMPLETE DATASET SETUP")
print("=" * 70)
print("\nThis will:")
print("  1. Download Kaggle datasets (Tomato, Bell Pepper, Chili)")
print("  2. Merge with existing Roboflow Chili data")
print("  3. Organize into train/val splits")
print("=" * 70)

# Step 1: Download Kaggle datasets
print("\n[1/4] Downloading Kaggle datasets...")
print("-" * 70)

print("  Downloading Tomato & Bell Pepper dataset...")
tomato_pepper_path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
print(f"    Downloaded to: {tomato_pepper_path}")

print("  Downloading Chili dataset 1...")
chili_path = kagglehub.dataset_download("ahmadalmahsiri/chili-plant-disease")
print(f"    Downloaded to: {chili_path}")

print("  Downloading Chili dataset 2...")
additional_chili_path = kagglehub.dataset_download("ravindubandara3002/chilli-plant-diseases-dataset")
print(f"    Downloaded to: {additional_chili_path}")

# Step 2: Define classes to keep
print("\n[2/4] Preparing class mappings...")
print("-" * 70)

tomato_pepper_classes = {
    'Tomato___healthy', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Bacterial_spot', 'Tomato___Leaf_Mold',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy'
}

chili_classes = ['Healthy', 'Leaf curl', 'Leaf spot', 'Whitefly', 'Yellowish']

print(f"  Tomato classes: 5")
print(f"  Bell Pepper classes: 2")
print(f"  Chili classes: 5")
print(f"  Total: 12 classes")

# Step 3: Create data directories (keep existing Roboflow data)
print("\n[3/4] Organizing dataset structure...")
print("-" * 70)

os.makedirs('data/train', exist_ok=True)
os.makedirs('data/val', exist_ok=True)

# Clean up empty folders first
print("  Cleaning up empty folders...")
for split in ['train', 'val']:
    split_dir = f'data/{split}'
    if os.path.exists(split_dir):
        for folder in os.listdir(split_dir):
            folder_path = os.path.join(split_dir, folder)
            if os.path.isdir(folder_path):
                # Count image files
                img_count = len([f for f in os.listdir(folder_path)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                if img_count == 0:
                    print(f"    Removing empty folder: {folder}")
                    shutil.rmtree(folder_path)

# Copy tomato and bell pepper classes
print("\n  Copying Tomato & Bell Pepper images...")
for split in ['train', 'valid']:
    source_dir = os.path.join(tomato_pepper_path, 'New Plant Diseases Dataset(Augmented)',
                               'New Plant Diseases Dataset(Augmented)', split)

    target_split = 'val' if split == 'valid' else split

    if os.path.exists(source_dir):
        for class_name in tomato_pepper_classes:
            src = os.path.join(source_dir, class_name)
            dst = os.path.join(f'data/{target_split}', class_name)

            if os.path.exists(src):
                print(f"    {split}/{class_name}...", end=" ")

                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)

                img_count = len([f for f in os.listdir(dst)
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"{img_count} images")

# Copy Kaggle chili classes
print("\n  Copying Kaggle Chili images (Dataset 1)...")
chili_splits = {'train': 'train', 'val': 'val'}

for chili_split, our_split in chili_splits.items():
    chili_split_dir = os.path.join(chili_path, 'Chili_Plant_Disease', chili_split)

    if os.path.exists(chili_split_dir):
        for class_name in chili_classes:
            # Try different naming variants
            for name_variant in [class_name, class_name.replace(' ', '_'),
                                class_name.replace(' ', '-'),
                                class_name.lower(), class_name.lower().replace(' ', '_')]:
                src = os.path.join(chili_split_dir, name_variant)

                if os.path.exists(src):
                    target_name = f'Chili___{class_name.replace(" ", "_")}'
                    dst = os.path.join(f'data/{our_split}', target_name)

                    print(f"    {chili_split}/{class_name}...", end=" ")

                    # Merge with existing Roboflow data if folder exists
                    if os.path.exists(dst):
                        # Copy files individually to merge
                        for img_file in os.listdir(src):
                            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                src_file = os.path.join(src, img_file)
                                dst_file = os.path.join(dst, f"kaggle1_{img_file}")
                                shutil.copy2(src_file, dst_file)
                    else:
                        shutil.copytree(src, dst)

                    img_count = len([f for f in os.listdir(dst)
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    print(f"{img_count} images (merged)")
                    break

# Copy additional Kaggle chili dataset
print("\n  Copying Kaggle Chili images (Dataset 2)...")
additional_chili_mapping = {
    'Healthy': 'Chilli___healthy',
    'Leaf curl': 'Chilli__Leaf_Curl_Virus',
    'Leaf spot': 'Chilli__Leaf_Spot',
    'Whitefly': 'Chilli __Whitefly',
    'Yellowish': 'Chilli __Yellowish'
}

chili_dataset_path = os.path.join(additional_chili_path, 'Chilli Plant Diseases Dataset(Augmented)',
                                   'Chilli Plant Diseases Dataset')

for our_class, dataset_class in additional_chili_mapping.items():
    for split, our_split in [('train', 'train'), ('valid', 'val')]:
        source_dir = os.path.join(chili_dataset_path, split, dataset_class)

        if os.path.exists(source_dir):
            target_name = f'Chili___{our_class.replace(" ", "_")}'
            target_dir = os.path.join(f'data/{our_split}', target_name)

            # Create target directory if it doesn't exist
            os.makedirs(target_dir, exist_ok=True)

            # Copy images (append to existing folder)
            img_copied = 0
            for img_file in os.listdir(source_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src = os.path.join(source_dir, img_file)
                    dst = os.path.join(target_dir, f"kaggle2_{img_file}")
                    shutil.copy2(src, dst)
                    img_copied += 1

            if img_copied > 0:
                total_imgs = len([f for f in os.listdir(target_dir)
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"    {split}/{our_class}: +{img_copied} images (total: {total_imgs})")

# Step 4: Show final statistics
print("\n[4/4] Dataset Summary")
print("=" * 70)

for split in ['train', 'val']:
    split_dir = f'data/{split}'
    if os.path.exists(split_dir):
        print(f"\n{split.upper()} Split:")
        print("-" * 70)

        classes = sorted(os.listdir(split_dir))
        total = 0

        # Group by plant type
        tomato_classes = []
        pepper_classes = []
        chili_classes = []

        for class_name in classes:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                count = len([f for f in os.listdir(class_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

                if count > 0:
                    if 'Tomato' in class_name:
                        tomato_classes.append((class_name, count))
                    elif 'Pepper' in class_name or 'bell' in class_name.lower():
                        pepper_classes.append((class_name, count))
                    elif 'Chili' in class_name:
                        chili_classes.append((class_name, count))
                    total += count

        if tomato_classes:
            print("  Tomato:")
            for name, count in tomato_classes:
                print(f"    {name}: {count:,} images")

        if pepper_classes:
            print("  Bell Pepper:")
            for name, count in pepper_classes:
                print(f"    {name}: {count:,} images")

        if chili_classes:
            print("  Chili:")
            for name, count in chili_classes:
                print(f"    {name}: {count:,} images")

        print(f"\n  TOTAL: {total:,} images")

print("\n" + "=" * 70)
print("[SUCCESS] Complete dataset setup finished!")
print("=" * 70)
print("\nYour dataset now includes:")
print("  - Tomato diseases (5 classes) from Kaggle")
print("  - Bell Pepper diseases (2 classes) from Kaggle")
print("  - Chili diseases (5 classes) from Kaggle + Roboflow")
print("\nYou can now run your training notebook!")
