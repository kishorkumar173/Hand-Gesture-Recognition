import os
import shutil
import random

# Paths
raw_dataset_dir = "dataset_raw"
train_dir = "dataset/train"
test_dir = "dataset/test"

# Split ratio
train_split = 0.8
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".png")

# Clear old dataset
if os.path.exists("dataset"):
    shutil.rmtree("dataset")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Go through each gesture class
for class_name in os.listdir(raw_dataset_dir):
    class_path = os.path.join(raw_dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # 🔥 Collect images from all subfolders
    files = []
    for root, _, filenames in os.walk(class_path):
        for f in filenames:
            if f.lower().endswith(IMG_EXTS):
                files.append(os.path.join(root, f))

    if not files:
        print(f"⚠️ No image files found in {class_path}, skipping.")
        continue

    # Shuffle and split
    random.shuffle(files)
    split_idx = int(len(files) * train_split)
    train_files = files[:split_idx]
    test_files = files[split_idx:]

    # Create class folders in train/test
    train_class_dir = os.path.join(train_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    # Copy files (rename to avoid duplicates)
    for i, f in enumerate(train_files):
        shutil.copy(f, os.path.join(train_class_dir, f"{class_name}_{i}.png"))
    for i, f in enumerate(test_files):
        shutil.copy(f, os.path.join(test_class_dir, f"{class_name}_{i}.png"))

    print(f"{class_name}: {len(train_files)} train, {len(test_files)} test")

print("✅ Split complete. Check 'dataset/train' and 'dataset/test'.")
