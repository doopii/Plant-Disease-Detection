"""
Convert Roboflow COCO Object Detection datasets to Classification format
and merge with existing Kaggle data.
"""

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

# Configuration
ROBOFLOW_DATASETS = {
    'ProjectTA': r'C:\Users\evago\Downloads\ProjectTA Chili Leaf Disease.v1i.coco',
    'SpotCurl': r'C:\Users\evago\Downloads\chilli.v1i.coco.spot_curl'
}

# Class name mapping: Roboflow name -> Standardized name
CLASS_MAPPING = {
    # Dataset 1: ProjectTA (Indonesian names)
    'Sehat': 'Chili___Healthy',
    'Keriting': 'Chili___Leaf_curl',
    'Bercak': 'Chili___Leaf_spot',
    'Kuning': 'Chili___Yellowish',
    'Whitefly': 'Chili___Whitefly',

    # Dataset 2: SpotCurl (English names)
    'leaf curl': 'Chili___Leaf_curl',
    'leaf spot': 'Chili___Leaf_spot',

    # Parent categories (skip these)
    'chili-plant-leaf-disease': None,
    'diseases': None
}

PROJECT_DATA_DIR = 'data'


def parse_coco_annotations(dataset_path, split='train'):
    """Parse COCO annotations and return image->classes mapping"""
    annotation_file = os.path.join(dataset_path, split, '_annotations.coco.json')

    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    # Build category ID -> name mapping
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Build image ID -> filename mapping
    images = {img['id']: img['file_name'] for img in coco_data['images']}

    # Build image -> set of classes mapping
    image_classes = defaultdict(set)
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        category_name = categories[ann['category_id']]

        # Map to standardized class name
        if category_name in CLASS_MAPPING:
            mapped_class = CLASS_MAPPING[category_name]
            if mapped_class:  # Skip None (parent categories)
                filename = images[image_id]
                image_classes[filename].add(mapped_class)

    return image_classes, os.path.join(dataset_path, split)


def copy_images_to_folders(image_classes, source_dir, target_base_dir, dataset_prefix):
    """Copy images from source to target folders organized by class"""
    stats = defaultdict(int)

    for image_filename, classes in image_classes.items():
        source_path = os.path.join(source_dir, image_filename)

        if not os.path.exists(source_path):
            print(f"Warning: Image not found: {source_path}")
            continue

        # For each class this image belongs to
        for class_name in classes:
            # Create class directory if it doesn't exist
            class_dir = os.path.join(target_base_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # Create unique filename with dataset prefix
            base_name, ext = os.path.splitext(image_filename)
            target_filename = f"{dataset_prefix}_{base_name}{ext}"
            target_path = os.path.join(class_dir, target_filename)

            # Copy image
            shutil.copy2(source_path, target_path)
            stats[class_name] += 1

    return stats


def process_all_datasets():
    """Process all Roboflow datasets and merge with existing data"""

    print("=" * 70)
    print("ROBOFLOW DATASET CONVERSION & MERGE")
    print("=" * 70)
    print()

    all_stats = {'train': defaultdict(int), 'val': defaultdict(int), 'test': defaultdict(int)}

    for dataset_name, dataset_path in ROBOFLOW_DATASETS.items():
        print(f"\nProcessing: {dataset_name}")
        print("-" * 70)

        # Process each split
        for split in ['train', 'valid', 'test']:
            split_dir = os.path.join(dataset_path, split)

            if not os.path.exists(split_dir):
                print(f"  {split}: Not found, skipping")
                continue

            # Map 'valid' to 'val' for our directory structure
            target_split = 'val' if split == 'valid' else split

            # Parse annotations
            try:
                image_classes, source_dir = parse_coco_annotations(dataset_path, split)

                # Copy images to appropriate folders
                target_dir = os.path.join(PROJECT_DATA_DIR, target_split)
                stats = copy_images_to_folders(
                    image_classes,
                    source_dir,
                    target_dir,
                    f"roboflow_{dataset_name}_{split}"
                )

                # Update overall stats
                for class_name, count in stats.items():
                    all_stats[target_split][class_name] += count

                print(f"  {split}: {len(image_classes)} images processed")

            except Exception as e:
                print(f"  {split}: Error - {e}")

    # Print summary
    print("\n" + "=" * 70)
    print("MERGE SUMMARY")
    print("=" * 70)

    for split in ['train', 'val', 'test']:
        if all_stats[split]:
            print(f"\n{split.upper()} Split:")
            for class_name in sorted(all_stats[split].keys()):
                print(f"  {class_name}: +{all_stats[split][class_name]} images")

    # Show updated totals
    print("\n" + "=" * 70)
    print("UPDATED DATASET TOTALS")
    print("=" * 70)

    for split in ['train', 'val']:
        split_dir = os.path.join(PROJECT_DATA_DIR, split)
        if os.path.exists(split_dir):
            print(f"\n{split.upper()} Split:")
            classes = sorted(os.listdir(split_dir))
            total = 0
            for class_name in classes:
                class_dir = os.path.join(split_dir, class_name)
                if os.path.isdir(class_dir):
                    count = len([f for f in os.listdir(class_dir)
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    print(f"  {class_name}: {count:,} images")
                    total += count
            print(f"  TOTAL: {total:,} images")


if __name__ == "__main__":
    # Verify source datasets exist
    print("Checking Roboflow datasets...")
    for name, path in ROBOFLOW_DATASETS.items():
        if os.path.exists(path):
            print(f"  [OK] {name}: Found")
        else:
            print(f"  [ERROR] {name}: NOT FOUND at {path}")
            print(f"    Please update the path in ROBOFLOW_DATASETS")

    print()
    response = input("Proceed with conversion and merge? (yes/no): ")

    if response.lower() in ['yes', 'y']:
        process_all_datasets()
        print("\n[SUCCESS] Conversion complete!")
    else:
        print("Operation cancelled.")
