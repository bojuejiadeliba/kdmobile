#!/usr/bin/env python3
"""
Script to create caltech_labels.json from the manually downloaded Caltech-256 dataset
"""

import os
import json


def create_caltech_labels():
    # Path to your 256_ObjectCategories folder
    caltech_path = "D:/AU/Dissertation/KLMobileProject/eval_datasets/caltech256/256_ObjectCategories"

    if not os.path.exists(caltech_path):
        print(f"Error: Caltech-256 dataset not found at {caltech_path}")
        print("Please check the path and make sure the dataset is extracted there.")
        return

    # Get all folder names (these are the class names)
    folders = [f for f in os.listdir(caltech_path) if os.path.isdir(os.path.join(caltech_path, f))]
    folders.sort()  # Sort to ensure consistent ordering

    print(f"Found {len(folders)} class folders")

    # Create mapping: class_name -> class_index
    meta_data = {}
    for idx, folder in enumerate(folders):
        # Remove number prefix (e.g., "001.ak47" -> "ak47")
        class_name = folder.split('.', 1)[1] if '.' in folder else folder
        # Replace hyphens/underscores with spaces for better prompts
        class_name = class_name.replace('-', ' ').replace('_', ' ')
        meta_data[class_name] = idx

    # Output path
    output_dir = "D:/AU/Dissertation/KLMobileProject/eval_datasets/caltech256"
    output_path = os.path.join(output_dir, "caltech_labels.json")

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(meta_data, f, indent=2)

    print(f"✓ Created {output_path} with {len(meta_data)} classes")

    # Show first few classes as example
    print("\nFirst 10 classes:")
    for i, (class_name, idx) in enumerate(meta_data.items()):
        if i < 10:
            print(f"  {idx:3d}: {class_name}")
        else:
            break
    print("  ...")

    return meta_data


if __name__ == "__main__":
    create_caltech_labels()