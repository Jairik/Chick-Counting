#!/usr/bin/env python3
import os
import glob
import random
import shutil

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_DIR = "/mnt/linuxlab/home/aforti2/PerdueFarms/RGB-USB Chick Labels 1.v1i.yolov11"
OUTPUT_BASE = "/mnt/linuxlab/home/aforti2/Chick-Counting/backend/chick-test-1"
VIDEO_TAGS  = ["vf1", "vf2", "vf3", "vf4", "vf5"]
TRAIN_SPLIT = 0.9    # fraction of down-sampled frames per video to use for training
SEED        = 12345  # for reproducibility
# ────────────────────────────────────────────────────────────────────────────────

random.seed(SEED)

# 1) Load original data.yaml
orig_yaml = os.path.join(DATASET_DIR, "data.yaml")
if not os.path.isfile(orig_yaml):
    raise FileNotFoundError(f"data.yaml not found in {DATASET_DIR!r}")

# 2) Group all (image, label) pairs by video prefix
pairs_by_vid = {tag: [] for tag in VIDEO_TAGS}
for img_path in glob.glob(os.path.join(DATASET_DIR, "train", "images", "*.jpg")):
    base = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(DATASET_DIR, "train", "labels", base + ".txt")
    if os.path.exists(lbl_path):
        prefix = base.split("_", 1)[0]
        if prefix in pairs_by_vid:
            pairs_by_vid[prefix].append((img_path, lbl_path))

# 3) Determine the minimum number of frames across all videos
counts = {tag: len(pairs_by_vid[tag]) for tag in VIDEO_TAGS}
min_count = min(counts.values())
print("Per‑video frame counts:", counts)
print(f"Down‑sampling each video to {min_count} frames for equal contribution.\n")

# 4) Build each fold
for i, holdout in enumerate(VIDEO_TAGS, start=1):
    fold_dir = os.path.join(OUTPUT_BASE, f"fold_{i}")
    print(f"Building fold_{i}: holding out {holdout}")

    # a) Create directories & copy data.yaml
    for split in ("train", "val"):
        os.makedirs(os.path.join(fold_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, split, "labels"), exist_ok=True)
    shutil.copy(orig_yaml, os.path.join(fold_dir, "data.yaml"))

    # b) Sample per‑video and split
    train_list = []
    val_list   = []
    for tag in VIDEO_TAGS:
        if tag == holdout:
            continue
        group = pairs_by_vid[tag]
        sampled = random.sample(group, min_count)
        cut = int(min_count * TRAIN_SPLIT)
        train_list.extend(sampled[:cut])
        val_list.extend(sampled[cut:])

    print(f"  Total train examples: {len(train_list)}")
    print(f"  Total val   examples: {len(val_list)}\n")

    # c) Copy files into fold directories
    for split_name, dataset in (("train", train_list), ("val", val_list)):
        for img_src, lbl_src in dataset:
            dst_img = os.path.join(fold_dir, split_name, "images", os.path.basename(img_src))
            dst_lbl = os.path.join(fold_dir, split_name, "labels", os.path.basename(lbl_src))
            shutil.copy(img_src, dst_img)
            shutil.copy(lbl_src, dst_lbl)

print("✅ All folds created with equal per‑video contributions.")
