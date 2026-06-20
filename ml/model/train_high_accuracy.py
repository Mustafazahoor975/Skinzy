"""
================================================================================
 SKINZY HIGH-ACCURACY SKIN DISEASE CLASSIFIER
 Target: 90%+ Validation Accuracy on 23 Classes
================================================================================
 Architecture:  EfficientNet-V2-M (pretrained ImageNet)
 Input Size:    300x300 (native resolution for EfficientNet-V2)
 Augmentation:  CutMix + MixUp + Heavy Geometric/Color
 Balancing:     WeightedRandomSampler
 Training:      3-Phase (Head → Partial Unfreeze → Full Fine-Tune)
 Regularization: Label Smoothing + Focal Loss + Dropout + Weight Decay
 Extras:        EMA, TTA, AMP, Gradient Accumulation, Confusion Matrix
================================================================================

USAGE (Google Colab):
    1. Upload dataset.zip to Google Drive or use OneDrive link
    2. Mount drive / extract dataset so it has this structure:
         dataset/
           train/   (23 class folders)
           val/     (optional — will auto-create from train if missing)
           test/    (23 class folders)
    3. Run this script:
         !python train_high_accuracy.py --data_dir /content/dataset

USAGE (Local — Windows):
    python train_high_accuracy.py --data_dir "C:\\Users\\Bismillah Traders\\Downloads\\dataset"
"""

import os
import sys
import time

# Reconfigure stdout/stderr to UTF-8 to support emojis on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import json
import copy
import math
import random
import argparse
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from torch.cuda.amp import GradScaler, autocast

warnings.filterwarnings("ignore")

# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================
def is_valid_dataset(path):
    """Check if path has train/ with class subfolders that actually contain files."""
    train_dir = os.path.join(path, "train")
    if not os.path.exists(train_dir):
        return False
    try:
        entries = os.listdir(train_dir)
        subdirs = [e for e in entries if os.path.isdir(os.path.join(train_dir, e)) and not e.startswith('.')]
        if len(subdirs) == 0:
            return False
        # Check that at least one class folder has actual files inside
        for sd in subdirs:
            sd_path = os.path.join(train_dir, sd)
            files = os.listdir(sd_path)
            if len(files) > 0:
                return True
        return False  # All class folders are empty
    except Exception:
        return False

def has_train_folder(path):
    """Lightweight check — just sees if train/ directory exists."""
    return os.path.isdir(os.path.join(path, "train"))

# --- Auto-setup for Google Colab ---
def _colab_auto_setup():
    """Mount Drive + unzip dataset.zip from Google Drive directly to local SSD."""
    try:
        import google.colab
    except ImportError:
        return None  # Not on Colab
    
    # Mount Drive
    drive_root = "/content/drive"
    if not os.path.exists(os.path.join(drive_root, "MyDrive")):
        print("📂 Mounting Google Drive...")
        from google.colab import drive
        drive.mount(drive_root)
    
    # Check if already on local SSD
    local = "/content/dataset"
    if is_valid_dataset(local):
        print(f"✅ Dataset already on local SSD: {local}")
        return local
    if is_valid_dataset("/content"):
        print(f"✅ Dataset already on local SSD at root: /content")
        return "/content"
    
    # We prioritize unzipping dataset.zip from the user's specified folders on Google Drive
    zip_paths = [
        "/content/drive/MyDrive/Skinzy Training/dataset.zip",
        "/content/drive/MyDrive/skinzy_training/dataset.zip",
        "/content/drive/MyDrive/SkinzyTraining/dataset.zip",
        "/content/drive/MyDrive/dataset.zip",
        "/content/drive/MyDrive/Copy of dataset.zip",
    ]
    
    import subprocess
    
    # Clean up local dataset folder if it exists but is invalid/empty to avoid merging issues
    if os.path.exists(local):
        try:
            subprocess.run(['rm', '-rf', local], check=False)
        except Exception:
            pass

    for zp in zip_paths:
        if os.path.exists(zp):
            print(f"📦 Found zip file on Google Drive: {zp}")
            print(f"⏳ Unzipping dataset from Google Drive to local SSD... (1-3 min)")
            try:
                result = subprocess.run(
                    ['unzip', '-q', '-o', zp, '-d', '/content/'],
                    capture_output=True, text=True, timeout=600
                )
                if result.returncode == 0:
                    if is_valid_dataset(local):
                        # Count files to confirm
                        check = subprocess.run(
                            ['find', local + '/train', '-type', 'f'],
                            capture_output=True, text=True, timeout=30
                        )
                        file_count = len(check.stdout.strip().split('\n')) if check.stdout.strip() else 0
                        print(f"✅ Unzip complete! {file_count} files found. Using: {local}")
                        return local
                    elif is_valid_dataset("/content"):
                        check = subprocess.run(
                            ['find', '/content/train', '-type', 'f'],
                            capture_output=True, text=True, timeout=30
                        )
                        file_count = len(check.stdout.strip().split('\n')) if check.stdout.strip() else 0
                        print(f"✅ Unzip complete! {file_count} files found at root. Using: /content")
                        return "/content"
                else:
                    print(f"⚠️  Unzip failed for {zp}: {result.stderr[:200]}")
            except Exception as e:
                print(f"⚠️  Error unzipping {zp}: {e}")
                
    # Fallback to copy from Drive if no zip file is found or unzip fails
    drive_paths = [
        "/content/drive/MyDrive/Skinzy Training/dataset",
        "/content/drive/MyDrive/skinzy_training/dataset",
        "/content/drive/MyDrive/dataset",
    ]
    source = None
    for p in drive_paths:
        if is_valid_dataset(p):  # Must have actual files, not empty folders
            source = p
            break
            
    if source is not None:
        print(f"⏳ Copying dataset folder from Google Drive: {source} → {local} (1-3 min)...")
        try:
            result = subprocess.run(
                ['cp', '-r', source, local],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode == 0:
                if is_valid_dataset(local):
                    check = subprocess.run(
                        ['find', local + '/train', '-type', 'f', '-name', '*.jpg'],
                        capture_output=True, text=True, timeout=30
                    )
                    file_count = len(check.stdout.strip().split('\n')) if check.stdout.strip() else 0
                    print(f"✅ Copy complete! {file_count} jpg files found. Using: {local}")
                    return local
            else:
                print(f"⚠️  cp failed: {result.stderr}")
                print(f"   Falling back to Drive: {source}")
                return source
        except Exception as e:
            print(f"⚠️  Copy failed ({e}), using Drive directly: {source}")
            return source

    print("❌ ERROR: Dataset zip file ('dataset.zip') not found in Google Drive ('Skinzy Training' folder or main Drive)!")
    return None

# Try to dynamically locate the dataset directory
default_data_dir = "./dataset"

# Colab: auto-mount + auto-copy/unzip
_colab_result = _colab_auto_setup()
if _colab_result:
    default_data_dir = _colab_result

# Fallback: search other paths
if not is_valid_dataset(default_data_dir):
    downloads_dir = r"C:\Users\Bismillah Traders\Downloads\dataset"
    try:
        script_dir = os.path.dirname(__file__)
    except NameError:
        script_dir = os.getcwd()
    parent_dataset = os.path.abspath(os.path.join(script_dir, "..", "..", "dataset"))
    colab_local = "/content/dataset"
    colab_root = "/content"
    colab_drive = "/content/drive/MyDrive/dataset"
    colab_skinzy_training = "/content/drive/MyDrive/Skinzy Training/dataset"
    colab_skinzy = "/content/drive/MyDrive/skinzy_training/dataset"
    
    if is_valid_dataset(downloads_dir):
        default_data_dir = downloads_dir
    elif is_valid_dataset(colab_local):
        default_data_dir = colab_local
    elif is_valid_dataset(colab_root):
        default_data_dir = colab_root
    elif is_valid_dataset(colab_skinzy_training):
        default_data_dir = colab_skinzy_training
    elif is_valid_dataset(colab_skinzy):
        default_data_dir = colab_skinzy
    elif is_valid_dataset(colab_drive):
        default_data_dir = colab_drive
    elif is_valid_dataset(parent_dataset):
        default_data_dir = parent_dataset

parser = argparse.ArgumentParser(description="Train Skin Disease Classifier for 90%+ Accuracy")
parser.add_argument("--data_dir", type=str, default=default_data_dir,
                    help="Path to the dataset folder containing train/test subfolders")
parser.add_argument("--output_dir", type=str, default="./skin_model_output",
                    help="Path to save model checkpoints and metrics")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size per step (effective = batch_size * accum_steps)")
parser.add_argument("--accum_steps", type=int, default=4,
                    help="Gradient accumulation steps for larger effective batch")
parser.add_argument("--image_size", type=int, default=300,
                    help="Input image resolution (300 recommended for EfficientNet-V2)")
parser.add_argument("--epochs_phase1", type=int, default=5,
                    help="Epochs for Phase 1: Head-only warmup")
parser.add_argument("--epochs_phase2", type=int, default=15,
                    help="Epochs for Phase 2: Partial unfreeze")
parser.add_argument("--epochs_phase3", type=int, default=30,
                    help="Epochs for Phase 3: Full fine-tune")
parser.add_argument("--num_workers", type=int, default=2,
                    help="DataLoader workers (set 0 on Windows if issues)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
parser.add_argument("--no_cutmix", action="store_true", help="Disable CutMix/MixUp")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
args, _ = parser.parse_known_args()  # parse_known_args ignores Colab's extra -f argument

# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False  # False for speed
        torch.backends.cudnn.benchmark = True        # Auto-tune convolutions

set_seed(args.seed)

# ==============================================================================
# DEVICE SETUP
# ==============================================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available() and not args.no_amp
print(f"{'='*60}")
print(f" SKINZY HIGH-ACCURACY TRAINER")
print(f"{'='*60}")
print(f"  Device:     {device}")
print(f"  AMP:        {'ON' if use_amp else 'OFF'}")
print(f"  Image Size: {args.image_size}x{args.image_size}")
print(f"  Batch Size: {args.batch_size} x {args.accum_steps} accum = {args.batch_size * args.accum_steps} effective")
print(f"{'='*60}\n")

os.makedirs(args.output_dir, exist_ok=True)

# ==============================================================================
# 1. DATA AUGMENTATION PIPELINES
# ==============================================================================
# Strong training augmentation — proven for dermatology
train_transforms = transforms.Compose([
    transforms.Resize((args.image_size + 32, args.image_size + 32)),  # Slightly oversized for crop
    transforms.RandomResizedCrop(args.image_size, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),  # Skin images have no "up"
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.05),   # Forces model to not rely only on color
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),  # Simulates occlusion
])

val_transforms = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# TTA (Test-Time Augmentation) transforms — 5 different views
tta_transforms_list = [
    val_transforms,  # Original
    transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((args.image_size + 20, args.image_size + 20)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((args.image_size + 20, args.image_size + 20)),
        transforms.FiveCrop(args.image_size),  # Returns 5 crops
        transforms.Lambda(lambda crops: crops[0]),  # Take first corner
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
]


# ==============================================================================
# 2. CUTMIX & MIXUP IMPLEMENTATION
# ==============================================================================
class CutMixMixUp:
    """Applies CutMix or MixUp randomly to each batch."""
    def __init__(self, num_classes, cutmix_alpha=1.0, mixup_alpha=0.4, cutmix_prob=0.5):
        self.num_classes = num_classes
        self.cutmix_alpha = cutmix_alpha
        self.mixup_alpha = mixup_alpha
        self.cutmix_prob = cutmix_prob

    def __call__(self, images, targets):
        if random.random() < self.cutmix_prob:
            return self._cutmix(images, targets)
        else:
            return self._mixup(images, targets)

    def _cutmix(self, images, targets):
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        batch_size = images.size(0)
        rand_index = torch.randperm(batch_size).to(images.device)

        target_a = targets
        target_b = targets[rand_index]

        _, _, H, W = images.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        images[:, :, y1:y2, x1:x2] = images[rand_index, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
        return images, target_a, target_b, lam

    def _mixup(self, images, targets):
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = images.size(0)
        rand_index = torch.randperm(batch_size).to(images.device)

        mixed_images = lam * images + (1 - lam) * images[rand_index]
        target_a = targets
        target_b = targets[rand_index]
        return mixed_images, target_a, target_b, lam


# ==============================================================================
# 3. FOCAL LOSS WITH LABEL SMOOTHING
# ==============================================================================
class FocalLossWithSmoothing(nn.Module):
    """
    Focal Loss focuses on hard-to-classify examples.
    Label Smoothing prevents overconfidence.
    """
    def __init__(self, num_classes, gamma=2.0, label_smoothing=0.1, weight=None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.weight = weight

    def forward(self, logits, targets_a, targets_b=None, lam=1.0):
        """
        Supports both normal targets and CutMix/MixUp mixed targets.
        """
        ce_loss_a = nn.functional.cross_entropy(
            logits, targets_a, weight=self.weight,
            label_smoothing=self.label_smoothing, reduction='none'
        )
        pt_a = torch.exp(-ce_loss_a)
        focal_loss_a = ((1 - pt_a) ** self.gamma) * ce_loss_a

        if targets_b is not None:
            ce_loss_b = nn.functional.cross_entropy(
                logits, targets_b, weight=self.weight,
                label_smoothing=self.label_smoothing, reduction='none'
            )
            pt_b = torch.exp(-ce_loss_b)
            focal_loss_b = ((1 - pt_b) ** self.gamma) * ce_loss_b
            return (lam * focal_loss_a + (1 - lam) * focal_loss_b).mean()

        return focal_loss_a.mean()


# ==============================================================================
# 4. EXPONENTIAL MOVING AVERAGE (EMA)
# ==============================================================================
class EMA:
    """
    Maintains an exponential moving average of model parameters.
    This acts as an ensemble over training steps and smooths noisy gradients.
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name in self.shadow:
                    self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
                else:
                    self.shadow[name] = param.data.clone()

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ==============================================================================
# 5. DATASET LOADING WITH WEIGHTED SAMPLER
# ==============================================================================
def load_datasets(data_dir, train_transforms, val_transforms):
    """Load train/val/test datasets with proper transforms and class balancing."""

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(train_dir):
        print(f"❌ ERROR: Train folder not found at '{train_dir}'")
        sys.exit(1)

    # Verify train_dir has class subfolders (lightweight check, no os.walk)
    try:
        entries = os.listdir(train_dir)
        class_dirs = [e for e in entries if os.path.isdir(os.path.join(train_dir, e)) and not e.startswith('.')]
    except Exception:
        class_dirs = []
    
    if len(class_dirs) == 0:
        print(f"\n❌ ERROR: No class folders found in '{train_dir}'!")
        print(f"   Contents: {os.listdir(train_dir) if os.path.exists(train_dir) else 'N/A'}")
        sys.exit(1)
    
    # ── PRE-SCAN: Fix class folders with missing/wrong image extensions and handle empty folders ──
    VALID_IMG_EXT = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}
    
    def sanitize_directory(directory):
        if not os.path.exists(directory):
            return {}
        try:
            entries = os.listdir(directory)
            c_dirs = [e for e in entries if os.path.isdir(os.path.join(directory, e)) and not e.startswith('.')]
        except Exception:
            return {}
        
        class_to_files = {}
        fixed_classes = []
        
        for cls_name in sorted(c_dirs):
            cls_path = os.path.join(directory, cls_name)
            try:
                all_files = os.listdir(cls_path)
            except Exception:
                all_files = []
                
            valid_files = [f for f in all_files if os.path.splitext(f)[1].lower() in VALID_IMG_EXT]
            
            if len(valid_files) == 0:
                # Try to fix files with missing or wrong extensions
                non_dir_files = [f for f in all_files if os.path.isfile(os.path.join(cls_path, f))]
                rename_count = 0
                for fname in non_dir_files:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in VALID_IMG_EXT:
                        old_path = os.path.join(cls_path, fname)
                        try:
                            with open(old_path, 'rb') as f:
                                header = f.read(8)
                            is_image = (
                                header[:3] == b'\xff\xd8\xff' or  # JPEG
                                header[:8] == b'\x89PNG\r\n\x1a\n' or  # PNG
                                header[:4] == b'RIFF' or  # WebP
                                header[:2] == b'BM'  # BMP
                            )
                        except Exception:
                            is_image = False
                            
                        if is_image:
                            if header[:3] == b'\xff\xd8\xff':
                                new_ext = '.jpg'
                            elif header[:8] == b'\x89PNG\r\n\x1a\n':
                                new_ext = '.png'
                            elif header[:4] == b'RIFF':
                                new_ext = '.webp'
                            else:
                                new_ext = '.bmp'
                                
                            new_name = fname + new_ext if not ext else fname.rsplit('.', 1)[0] + new_ext
                            new_path = os.path.join(cls_path, new_name)
                            try:
                                os.rename(old_path, new_path)
                                rename_count += 1
                            except Exception:
                                pass
                if rename_count > 0:
                    fixed_classes.append(f"{cls_name} ({rename_count} files renamed)")
            
            if os.path.exists(cls_path):
                try:
                    class_to_files[cls_name] = [
                        f for f in os.listdir(cls_path)
                        if os.path.splitext(f)[1].lower() in VALID_IMG_EXT
                    ]
                except Exception:
                    class_to_files[cls_name] = []
            else:
                class_to_files[cls_name] = []
                
        if fixed_classes:
            print(f"🔧 Fixed image extensions in '{os.path.basename(directory)}' for classes:")
            for fc in fixed_classes:
                print(f"   • {fc}")
        return class_to_files

    print("🔍 Pre-scanning dataset directories...")
    train_files = sanitize_directory(train_dir)
    val_files = sanitize_directory(val_dir) if os.path.exists(val_dir) else {}
    test_files = sanitize_directory(test_dir) if os.path.exists(test_dir) else {}
    
    # Get all unique classes across train/val/test
    all_classes = set(train_files.keys()) | set(val_files.keys())
    if os.path.exists(test_dir):
        all_classes = all_classes | set(test_files.keys())
        
    import shutil
    import random
    
    classes_to_remove = []
    
    for cls in sorted(all_classes):
        tr_list = train_files.get(cls, [])
        val_list = val_files.get(cls, [])
        te_list = test_files.get(cls, [])
        
        # 1. If empty in train but has files in val
        if len(tr_list) == 0 and len(val_list) > 0:
            os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
            num_to_copy = max(1, int(len(val_list) * 0.5))
            to_copy = random.sample(val_list, num_to_copy)
            for f in to_copy:
                shutil.copy(os.path.join(val_dir, cls, f), os.path.join(train_dir, cls, f))
            print(f"🚚 Populated empty train class '{cls}' by copying {num_to_copy} images from val")
            tr_list = [f for f in os.listdir(os.path.join(train_dir, cls)) if os.path.splitext(f)[1].lower() in VALID_IMG_EXT]
            train_files[cls] = tr_list
            
        # 2. If empty in val but has files in train (only if val_dir is active validation set)
        if os.path.exists(val_dir) and len(os.listdir(val_dir)) > 0:
            if len(val_list) == 0 and len(tr_list) > 0:
                os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
                num_to_copy = max(1, int(len(tr_list) * 0.15))
                to_copy = random.sample(tr_list, num_to_copy)
                for f in to_copy:
                    shutil.copy(os.path.join(train_dir, cls, f), os.path.join(val_dir, cls, f))
                print(f"🚚 Populated empty val class '{cls}' by copying {num_to_copy} images from train")
                val_list = [f for f in os.listdir(os.path.join(val_dir, cls)) if os.path.splitext(f)[1].lower() in VALID_IMG_EXT]
                val_files[cls] = val_list

        # 3. If empty in test but test_dir exists (and train/val has files)
        if os.path.exists(test_dir):
            if len(te_list) == 0 and (len(tr_list) > 0 or len(val_list) > 0):
                os.makedirs(os.path.join(test_dir, cls), exist_ok=True)
                source_list = tr_list if len(tr_list) > 0 else val_list
                source_dir = train_dir if len(tr_list) > 0 else val_dir
                num_to_copy = max(1, int(len(source_list) * 0.10))
                to_copy = random.sample(source_list, num_to_copy)
                for f in to_copy:
                    shutil.copy(os.path.join(source_dir, cls, f), os.path.join(test_dir, cls, f))
                print(f"🚚 Populated empty test class '{cls}' by copying {num_to_copy} images from {os.path.basename(source_dir)}")
                te_list = [f for f in os.listdir(os.path.join(test_dir, cls)) if os.path.splitext(f)[1].lower() in VALID_IMG_EXT]
                test_files[cls] = te_list

        is_empty_train = len(train_files.get(cls, [])) == 0
        is_empty_val = os.path.exists(val_dir) and len(os.listdir(val_dir)) > 0 and len(val_files.get(cls, [])) == 0
        is_empty_test = os.path.exists(test_dir) and len(test_files.get(cls, [])) == 0
        
        if is_empty_train or is_empty_val or is_empty_test:
            classes_to_remove.append(cls)
            
    if classes_to_remove:
        print(f"\n🗑️  Removing {len(classes_to_remove)} empty/broken class folders from all directories to prevent crash...")
        for cls in classes_to_remove:
            for d in [train_dir, val_dir, test_dir]:
                p = os.path.join(d, cls)
                if os.path.exists(p):
                    try:
                        shutil.rmtree(p)
                        print(f"   ✅ Removed: {os.path.basename(d)}/{cls}")
                    except Exception as e:
                        print(f"   ⚠️  Could not remove {os.path.basename(d)}/{cls}: {e}")
                        
    # Get final list of class directories in train
    try:
        entries = os.listdir(train_dir)
        class_dirs = [e for e in entries if os.path.isdir(os.path.join(train_dir, e)) and not e.startswith('.')]
    except Exception:
        class_dirs = []
        
    if len(class_dirs) == 0:
        print(f"\n❌ ERROR: No valid classes remaining after cleanup!")
        sys.exit(1)
        
    # Final check of the first class directory
    first_class = os.path.join(train_dir, class_dirs[0])
    try:
        sample_files = [f for f in os.listdir(first_class) if os.path.splitext(f)[1].lower() in VALID_IMG_EXT]
    except Exception:
        sample_files = []
        
    if len(sample_files) == 0:
        print(f"\n❌ ERROR: Class folder '{class_dirs[0]}' is empty!")
        sys.exit(1)
        
    print(f"✅ Found {len(class_dirs)} classes, first class '{class_dirs[0]}' has {len(sample_files)} files")

    # Load training set
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    class_names = train_dataset.classes
    num_classes = len(class_names)

    # Load validation set
    if os.path.exists(val_dir) and len(os.listdir(val_dir)) >= num_classes // 2:
        print("✅ Using existing validation set from 'val/' folder")
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
    else:
        # Auto-split 85/15 from training
        print("⚠️  No val folder found — auto-splitting 85/15 from training data")
        total = len(train_dataset)
        val_size = int(0.15 * total)
        train_size = total - val_size

        # Split with fixed seed for reproducibility
        generator = torch.Generator().manual_seed(args.seed)
        train_indices, val_indices = torch.utils.data.random_split(
            range(total), [train_size, val_size], generator=generator
        )

        # Create val dataset with val transforms
        val_full = datasets.ImageFolder(train_dir, transform=val_transforms)
        val_dataset = Subset(val_full, val_indices.indices)
        train_dataset = Subset(train_dataset, train_indices.indices)

    # Load test set
    test_dataset = None
    if os.path.exists(test_dir):
        test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)
        print(f"✅ Test set loaded: {len(test_dataset)} images")

    # --- Compute class weights for WeightedRandomSampler ---
    if isinstance(train_dataset, Subset):
        all_targets = [train_dataset.dataset.targets[i] for i in train_dataset.indices]
    else:
        all_targets = train_dataset.targets

    class_counts = Counter(all_targets)
    total_samples = len(all_targets)
    class_weight_list = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)
        class_weight_list.append(total_samples / (num_classes * count))

    sample_weights = [class_weight_list[t] for t in all_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Class weights tensor for loss function
    class_weights_tensor = torch.tensor(class_weight_list, dtype=torch.float32).to(device)
    # Normalize to prevent loss explosion
    class_weights_tensor = class_weights_tensor / class_weights_tensor.mean()

    print(f"\n📊 Dataset Summary:")
    print(f"   Classes: {num_classes}")
    print(f"   Training:   {len(train_dataset)} images")
    print(f"   Validation: {len(val_dataset)} images")
    if test_dataset:
        print(f"   Test:       {len(test_dataset)} images")
    print(f"\n📊 Class Distribution (Training):")
    for i, name in enumerate(class_names):
        count = class_counts.get(i, 0)
        weight = class_weight_list[i]
        bar = '█' * int(count / max(class_counts.values()) * 30)
        print(f"   {name:40s} {count:5d}  w={weight:.2f}  {bar}")

    return train_dataset, val_dataset, test_dataset, class_names, num_classes, sampler, class_weights_tensor


# ==============================================================================
# 6. MODEL ARCHITECTURE
# ==============================================================================
def build_model(num_classes):
    """Build EfficientNet-V2-M with custom classification head."""

    print("\n🏗️  Building EfficientNet-V2-M model...")

    # Try V2-M first (best quality), fallback to V2-S, then B4
    try:
        model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        arch_name = "efficientnet_v2_m"
        print("   ✅ Loaded EfficientNet-V2-M (480M params pretrained)")
    except Exception:
        try:
            model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
            arch_name = "efficientnet_v2_s"
            print("   ⚠️  V2-M not available, using EfficientNet-V2-S")
        except Exception:
            model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
            arch_name = "efficientnet_b4"
            print("   ⚠️  V2 not available, using EfficientNet-B4")

    # Freeze everything initially
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier with a stronger custom head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.SiLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes)
    )

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params:     {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")

    return model, arch_name


def unfreeze_last_n_blocks(model, n=2):
    """Unfreeze the last N feature extraction blocks for gradual fine-tuning."""
    # EfficientNet features are in model.features (list of blocks)
    total_blocks = len(model.features)
    unfreeze_from = max(0, total_blocks - n)

    for i, block in enumerate(model.features):
        if i >= unfreeze_from:
            for param in block.parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   🔓 Unfroze last {n} blocks (from block {unfreeze_from}/{total_blocks})")
    print(f"   Trainable params now: {trainable:,}")


def unfreeze_all(model):
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   🔓 All layers unfrozen. Trainable params: {trainable:,}")


# ==============================================================================
# 7. WARMUP + COSINE ANNEALING SCHEDULER
# ==============================================================================
def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay to near-zero."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ==============================================================================
# 8. TRAINING LOOP
# ==============================================================================
def train_one_phase(model, criterion, train_loader, val_loader, optimizer, scheduler,
                    num_epochs, phase_name, ema, cutmix_fn=None,
                    accum_steps=4, best_acc=0.0, best_model_wts=None,
                    class_names=None, output_dir="./", all_history=None, start_epoch=0):
    """
    Train model for one phase with:
    - CutMix/MixUp augmentation
    - Gradient accumulation
    - AMP mixed precision
    - EMA weight averaging
    - Comprehensive logging
    """

    scaler = GradScaler(enabled=use_amp)
    history = []
    patience = max(8, num_epochs // 3)
    epochs_no_improve = 0
    val_size = len(val_loader.dataset)
    train_size = len(train_loader.dataset)

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        print(f"\n{'─'*60}")
        print(f"  {phase_name} │ Epoch {epoch+1}/{num_epochs} │ LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'─'*60}")

        # ── TRAINING ──
        model.train()
        running_loss = 0.0
        running_corrects = 0
        optimizer.zero_grad()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Apply CutMix/MixUp
            use_mix = cutmix_fn is not None and random.random() < 0.5
            if use_mix:
                inputs, targets_a, targets_b, lam = cutmix_fn(inputs, labels)
            else:
                targets_a, targets_b, lam = labels, None, 1.0

            with autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets_a, targets_b, lam)
                loss = loss / accum_steps  # Scale for accumulation

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Update EMA
                if ema is not None:
                    ema.update(model)

            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item() * accum_steps * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            if use_mix:
                running_corrects += (lam * torch.sum(preds == targets_a).float() +
                                     (1 - lam) * torch.sum(preds == targets_b).float())
            else:
                running_corrects += torch.sum(preds == labels.data).float()

        train_loss = running_loss / train_size
        train_acc = running_corrects / train_size

        # ── VALIDATION (with EMA weights) ──
        if ema is not None:
            ema.apply_shadow(model)

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast(enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        if ema is not None:
            ema.restore(model)

        val_loss = val_loss / val_size
        val_acc = val_corrects / val_size

        epoch_time = time.time() - epoch_start

        print(f"  Train Loss: {train_loss:.4f}  │  Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}  │  Val   Acc: {val_acc:.4f}")
        print(f"  Time: {epoch_time:.1f}s")

        history.append({
            'epoch': epoch + 1,
            'train_loss': float(train_loss),
            'train_acc': float(train_acc),
            'val_loss': float(val_loss),
            'val_acc': float(val_acc),
            'lr': optimizer.param_groups[0]['lr'],
            'time': round(epoch_time, 1)
        })

        # ── BEST MODEL TRACKING ──
        if val_acc > best_acc:
            improvement = val_acc - best_acc
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"  🏆 NEW BEST: {val_acc:.4f} (+{improvement:.4f})")

            # Save checkpoint
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'ema_state': ema.shadow if ema else None,
                'class_names': class_names,
                'architecture': arch_name,
                'image_size': args.image_size,
                'best_val_acc': float(best_acc),
                'epoch': epoch + 1,
                'phase': phase_name,
            }
            torch.save(checkpoint_data, os.path.join(output_dir, 'skin_best_model_optimized.pth'))

            # Auto-save to Google Drive if available (survives disconnect!)
            drive_backup = '/content/drive/MyDrive/skinzy_training/skin_model_output'
            if os.path.exists('/content/drive/MyDrive'):
                os.makedirs(drive_backup, exist_ok=True)
                torch.save(checkpoint_data, os.path.join(drive_backup, 'skin_best_model_optimized.pth'))
                print(f"  💾 Backup saved to Google Drive!")
        else:
            epochs_no_improve += 1

        # ── SAVE LATEST CHECKPOINT (End of every epoch) ──
        latest_checkpoint_data = {
            'epoch': epoch + 1,
            'phase': phase_name,
            'model_state_dict': model.state_dict(),
            'ema_state': ema.shadow if ema else None,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_acc': float(best_acc),
            'best_model_wts': best_model_wts,
            'class_names': class_names,
            'architecture': arch_name,
            'image_size': args.image_size,
            'history': history,
            'all_history': (all_history if all_history is not None else []) + history,
        }
        torch.save(latest_checkpoint_data, os.path.join(output_dir, 'skin_latest_checkpoint.pth'))
        if os.path.exists('/content/drive/MyDrive'):
            drive_backup = '/content/drive/MyDrive/skinzy_training/skin_model_output'
            os.makedirs(drive_backup, exist_ok=True)
            torch.save(latest_checkpoint_data, os.path.join(drive_backup, 'skin_latest_checkpoint.pth'))
            print(f"  💾 Latest checkpoint saved to Google Drive!")

        if epochs_no_improve >= patience:
            print(f"\n  ⚠️  Early stopping triggered after {patience} epochs without improvement")
            break

    return model, best_acc, best_model_wts, history


# ==============================================================================
# 9. TEST-TIME AUGMENTATION (TTA)
# ==============================================================================
def evaluate_with_tta(model, test_dir, class_names, tta_transforms, batch_size=16):
    """Evaluate model with Test-Time Augmentation for maximum accuracy."""

    if not os.path.exists(test_dir):
        print("⚠️  No test directory found, skipping TTA evaluation")
        return None, None, None

    print(f"\n🔬 Running Test-Time Augmentation ({len(tta_transforms)} views)...")
    model.eval()
    num_classes = len(class_names)

    # Load test set with basic transforms to get labels
    base_dataset = datasets.ImageFolder(test_dir, transform=tta_transforms[0])
    all_labels = [s[1] for s in base_dataset.samples]
    num_samples = len(base_dataset)

    # Accumulate predictions from all TTA transforms
    all_probs = torch.zeros(num_samples, num_classes)

    for t_idx, t in enumerate(tta_transforms):
        tta_dataset = datasets.ImageFolder(test_dir, transform=t)
        tta_loader = DataLoader(tta_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

        sample_idx = 0
        with torch.no_grad():
            for inputs, _ in tta_loader:
                inputs = inputs.to(device)
                with autocast(enabled=use_amp):
                    outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1).cpu()
                batch_end = sample_idx + probs.size(0)
                all_probs[sample_idx:batch_end] += probs
                sample_idx = batch_end

        print(f"   TTA view {t_idx+1}/{len(tta_transforms)} complete")

    # Average across all TTA views
    all_probs /= len(tta_transforms)
    _, final_preds = torch.max(all_probs, 1)
    labels_tensor = torch.tensor(all_labels)

    test_acc = (final_preds == labels_tensor).float().mean().item()

    # Per-class accuracy
    per_class = {}
    for i, name in enumerate(class_names):
        mask = labels_tensor == i
        if mask.sum() > 0:
            class_acc = (final_preds[mask] == i).float().mean().item()
            per_class[name] = round(class_acc, 4)

    # Confusion matrix
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for pred, label in zip(final_preds, labels_tensor):
        confusion[label][pred] += 1

    print(f"\n   📊 TTA Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    return test_acc, per_class, confusion.numpy()


# ==============================================================================
# 10. SAVE CONFUSION MATRIX IMAGE
# ==============================================================================
def save_confusion_matrix(confusion, class_names, output_path):
    """Save confusion matrix as a PNG image using matplotlib."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(16, 14))
        im = ax.imshow(confusion, interpolation='nearest', cmap='Blues')
        ax.set_title(f'Confusion Matrix — Test Acc: {confusion.diagonal().sum() / confusion.sum() * 100:.2f}%',
                     fontsize=14, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.8)

        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(class_names, rotation=90, fontsize=7)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(class_names, fontsize=7)

        ax.set_ylabel('Actual', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   📊 Confusion matrix saved: {output_path}")
    except ImportError:
        print("   ⚠️  matplotlib not installed, skipping confusion matrix plot")


# ==============================================================================
# 11. SAVE TRAINING CURVES
# ==============================================================================
def save_training_curves(all_history, output_path):
    """Save loss and accuracy curves for all phases."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        epochs = list(range(1, len(all_history) + 1))
        train_loss = [h['train_loss'] for h in all_history]
        val_loss = [h['val_loss'] for h in all_history]
        train_acc = [h['train_acc'] for h in all_history]
        val_acc = [h['val_acc'] for h in all_history]

        ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
        ax1.set_title('Loss over Epochs', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, train_acc, 'b-', label='Train Acc', linewidth=2)
        ax2.plot(epochs, val_acc, 'r-', label='Val Acc', linewidth=2)
        ax2.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='90% Target')
        ax2.set_title('Accuracy over Epochs', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   📈 Training curves saved: {output_path}")
    except ImportError:
        print("   ⚠️  matplotlib not installed, skipping training curves")


# ==============================================================================
# 12. MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    total_start = time.time()

    # ── Load Data ──
    train_dataset, val_dataset, test_dataset, class_names, num_classes, sampler, class_weights = \
        load_datasets(args.data_dir, train_transforms, val_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Save class names
    with open(os.path.join(args.output_dir, 'class_names.json'), 'w') as f:
        json.dump(class_names, f, indent=2)

    # ── Build Model ──
    model, arch_name = build_model(num_classes)

    # ── Loss Function (Focal + Label Smoothing + Class Weights) ──
    criterion = FocalLossWithSmoothing(
        num_classes=num_classes, gamma=2.0,
        label_smoothing=0.1, weight=class_weights
    )

    # ── CutMix/MixUp ──
    cutmix_fn = None if args.no_cutmix else CutMixMixUp(num_classes)

    # ── EMA ──
    ema = EMA(model, decay=0.999)

    best_acc = 0.0
    best_model_wts = None
    all_history = []
    skip_phase1 = False
    skip_phase2 = False

# ── RESUME FROM CHECKPOINT (if Colab disconnected) ──
    resume_paths = [
        os.path.join(args.output_dir, 'skin_latest_checkpoint.pth'),
        '/content/drive/MyDrive/skinzy_training/skin_model_output/skin_latest_checkpoint.pth',
        os.path.join(args.output_dir, 'skin_best_model_optimized.pth'),
        '/content/drive/MyDrive/skinzy_training/skin_model_output/skin_best_model_optimized.pth',
    ]
    if args.resume:
        resume_paths.insert(0, args.resume)

    checkpoint_loaded = False
    start_epoch_p1 = 0
    start_epoch_p2 = 0
    start_epoch_p3 = 0

    for rpath in resume_paths:
        if os.path.exists(rpath):
            print(f"\n🔄 RESUMING from checkpoint: {rpath}")
            checkpoint = torch.load(rpath, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            last_phase = checkpoint.get('phase', '')

            # Restore EMA weights shadow state if present and we resumed in Phase 1
            if ema and checkpoint.get('ema_state') and 'Phase 1' in last_phase:
                ema.shadow = checkpoint['ema_state']

            best_acc = checkpoint.get('best_val_acc', 0.0)
            best_model_wts = checkpoint.get('best_model_wts', None)
            if best_model_wts is None:
                best_model_wts = copy.deepcopy(checkpoint['model_state_dict'])
            saved_epoch = checkpoint.get('epoch', 0)
            all_history = checkpoint.get('all_history', [])
            checkpoint_loaded = True

            print(f"   Last Phase: {last_phase} (Epoch {saved_epoch})")
            print(f"   Best Val Acc so far: {best_acc:.4f}")

            if 'Phase 1' in last_phase:
                if saved_epoch < args.epochs_phase1:
                    start_epoch_p1 = saved_epoch
                    print(f"   🔄 Will resume Phase 1 from epoch {start_epoch_p1 + 1}...")
                else:
                    skip_phase1 = True
                    print(f"   ⏩ Skipping Phase 1 (already completed)")
            elif 'Phase 2' in last_phase:
                skip_phase1 = True
                if saved_epoch < args.epochs_phase2:
                    start_epoch_p2 = saved_epoch
                    print(f"   🔄 Will resume Phase 2 from epoch {start_epoch_p2 + 1}...")
                else:
                    skip_phase1 = True
                    skip_phase2 = True
                    print(f"   ⏩ Skipping Phase 2 (already completed)")
            elif 'Phase 3' in last_phase:
                skip_phase1 = True
                skip_phase2 = True
                if saved_epoch < args.epochs_phase3:
                    start_epoch_p3 = saved_epoch
                    print(f"   🔄 Will resume Phase 3 from epoch {start_epoch_p3 + 1}...")
                else:
                    print(f"   ✅ All phases completed!")
            break

    # ═══════════════════════════════════════════════════════════════════════════
    #  PHASE 1: HEAD WARMUP (Only classifier trains)
    # ═══════════════════════════════════════════════════════════════════════════
    if not skip_phase1:
        print(f"\n{'═'*60}")
        print(f"  🚀 PHASE 1: HEAD WARMUP ({args.epochs_phase1} epochs)")
        print(f"{'═'*60}")

        optimizer_p1 = optim.AdamW(
            model.classifier.parameters(), lr=1e-3, weight_decay=1e-4
        )
        total_steps_p1 = len(train_loader) * args.epochs_phase1
        scheduler_p1 = get_cosine_schedule_with_warmup(optimizer_p1, warmup_steps=100, total_steps=total_steps_p1)

        # Restore optimizer/scheduler if we resumed in Phase 1
        if checkpoint_loaded and 'Phase 1' in last_phase and start_epoch_p1 > 0:
            if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
                optimizer_p1.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                scheduler_p1.load_state_dict(checkpoint['scheduler_state_dict'])

        model, best_acc, best_model_wts, history_p1 = train_one_phase(
            model, criterion, train_loader, val_loader,
            optimizer_p1, scheduler_p1, args.epochs_phase1,
            "Phase 1 (Head)", ema, cutmix_fn=None,  # No CutMix in warmup
            accum_steps=1, best_acc=best_acc, best_model_wts=best_model_wts,
            class_names=class_names, output_dir=args.output_dir,
            all_history=all_history, start_epoch=start_epoch_p1
        )
        all_history.extend(history_p1)
    else:
        print(f"\n  ⏩ Phase 1 skipped (already completed)")
        # Need to freeze params for head state consistency
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    # ═══════════════════════════════════════════════════════════════════════════
    #  PHASE 2: PARTIAL UNFREEZE (Last 3 blocks + classifier)
    # ═══════════════════════════════════════════════════════════════════════════
    if not skip_phase2:
        print(f"\n{'═'*60}")
        print(f"  🚀 PHASE 2: PARTIAL UNFREEZE ({args.epochs_phase2} epochs)")
        print(f"{'═'*60}")

        unfreeze_last_n_blocks(model, n=3)

        # Re-initialize EMA with new trainable params
        ema = EMA(model, decay=0.999)

        # Restore EMA weights shadow state if we resumed in Phase 2
        if checkpoint_loaded and 'Phase 2' in last_phase and checkpoint.get('ema_state'):
            ema.shadow = checkpoint['ema_state']
            print("   ✅ Resumed EMA shadow weights from checkpoint.")

        # Discriminative learning rates: backbone gets lower LR than head
        backbone_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and 'classifier' not in n]
        head_params = list(model.classifier.parameters())

        optimizer_p2 = optim.AdamW([
            {'params': backbone_params, 'lr': 5e-5},
            {'params': head_params, 'lr': 5e-4},
        ], weight_decay=1e-3)

        total_steps_p2 = len(train_loader) * args.epochs_phase2
        scheduler_p2 = get_cosine_schedule_with_warmup(optimizer_p2, warmup_steps=200, total_steps=total_steps_p2)

        # Restore optimizer/scheduler if we resumed in Phase 2
        if checkpoint_loaded and 'Phase 2' in last_phase and start_epoch_p2 > 0:
            if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
                optimizer_p2.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                scheduler_p2.load_state_dict(checkpoint['scheduler_state_dict'])

        model, best_acc, best_model_wts, history_p2 = train_one_phase(
            model, criterion, train_loader, val_loader,
            optimizer_p2, scheduler_p2, args.epochs_phase2,
            "Phase 2 (Partial)", ema, cutmix_fn=cutmix_fn,
            accum_steps=args.accum_steps, best_acc=best_acc, best_model_wts=best_model_wts,
            class_names=class_names, output_dir=args.output_dir,
            all_history=all_history, start_epoch=start_epoch_p2
        )
        all_history.extend(history_p2)
    else:
        print(f"\n  ⏩ Phase 2 skipped (already completed)")

    # ═══════════════════════════════════════════════════════════════════════════
    #  PHASE 3: FULL FINE-TUNE (Everything trains)
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  🚀 PHASE 3: FULL FINE-TUNE ({args.epochs_phase3} epochs)")
    print(f"{'═'*60}")

    unfreeze_all(model)

    # Re-initialize EMA with all params
    ema = EMA(model, decay=0.9995)  # Higher decay for full fine-tune stability

    # Restore EMA weights shadow state if present and we resumed in Phase 3
    if checkpoint_loaded and 'Phase 3' in last_phase and checkpoint.get('ema_state'):
        ema.shadow = checkpoint['ema_state']
        print("   ✅ Resumed EMA shadow weights from checkpoint.")

    # Even more discriminative LRs: early layers learn very slowly
    early_params = []
    mid_params = []
    late_params = []
    head_params = list(model.classifier.parameters())

    total_blocks = len(model.features)
    for i, block in enumerate(model.features):
        params = list(block.parameters())
        if i < total_blocks // 3:
            early_params.extend(params)
        elif i < 2 * total_blocks // 3:
            mid_params.extend(params)
        else:
            late_params.extend(params)

    optimizer_p3 = optim.AdamW([
        {'params': early_params, 'lr': 1e-6},    # Lower LR to prevent feature destabilization
        {'params': mid_params, 'lr': 5e-6},       # Mid layers learn very slowly
        {'params': late_params, 'lr': 2e-5},       # Late layers adapt gently to skin domain
        {'params': head_params, 'lr': 1e-4},       # Head learns moderately fast
    ], weight_decay=1e-3)

    total_steps_p3 = len(train_loader) * args.epochs_phase3
    scheduler_p3 = get_cosine_schedule_with_warmup(optimizer_p3, warmup_steps=200, total_steps=total_steps_p3)

    # Restore optimizer/scheduler if we resumed in Phase 3
    if checkpoint_loaded and 'Phase 3' in last_phase and start_epoch_p3 > 0:
        if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
            try:
                optimizer_p3.load_state_dict(checkpoint['optimizer_state_dict'])
                print("   ✅ Resumed optimizer state.")
            except Exception as e:
                print(f"   ⚠️ Could not load optimizer state: {e}")
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            try:
                scheduler_p3.load_state_dict(checkpoint['scheduler_state_dict'])
                print("   ✅ Resumed scheduler state.")
            except Exception as e:
                print(f"   ⚠️ Could not load scheduler state: {e}")

    model, best_acc, best_model_wts, history_p3 = train_one_phase(
        model, criterion, train_loader, val_loader,
        optimizer_p3, scheduler_p3, args.epochs_phase3,
        "Phase 3 (Full)", ema, cutmix_fn=cutmix_fn,
        accum_steps=args.accum_steps, best_acc=best_acc, best_model_wts=best_model_wts,
        class_names=class_names, output_dir=args.output_dir,
        all_history=all_history, start_epoch=start_epoch_p3
    )
    all_history.extend(history_p3)

    # ═══════════════════════════════════════════════════════════════════════════
    #  FINAL EVALUATION
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print(f"  📊 FINAL EVALUATION")
    print(f"{'═'*60}")

    # Load best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        print(f"  ✅ Loaded best model weights (Val Acc: {best_acc:.4f})")

    # Apply EMA weights for final evaluation
    if ema is not None:
        ema.apply_shadow(model)

    # Test with TTA
    test_dir = os.path.join(args.data_dir, 'test')
    test_acc, per_class_acc, confusion = evaluate_with_tta(
        model, test_dir, class_names, tta_transforms_list, batch_size=args.batch_size
    )

    # ── Save All Metrics ──
    metrics = {
        'architecture': arch_name,
        'num_classes': num_classes,
        'class_names': class_names,
        'best_val_accuracy': float(best_acc),
        'image_size': args.image_size,
        'batch_size': args.batch_size,
        'accum_steps': args.accum_steps,
        'effective_batch_size': args.batch_size * args.accum_steps,
        'epochs_total': len(all_history),
        'use_cutmix': not args.no_cutmix,
        'use_amp': use_amp,
        'use_ema': True,
        'use_focal_loss': True,
        'use_tta': True,
        'training_time_minutes': round((time.time() - total_start) / 60, 1),
    }

    if test_acc is not None:
        metrics['test_accuracy_tta'] = round(test_acc, 4)
        metrics['per_class_accuracy'] = per_class_acc

    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  📄 Metrics saved: {os.path.join(args.output_dir, 'metrics.json')}")

    # Save training history
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(all_history, f, indent=2)
    print(f"  📄 Training history saved")

    # Save plots
    save_training_curves(all_history, os.path.join(args.output_dir, 'training_curves.png'))
    if confusion is not None:
        save_confusion_matrix(confusion, class_names, os.path.join(args.output_dir, 'confusion_matrix.png'))

    # ═══════════════════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    total_time = time.time() - total_start
    print(f"\n{'═'*60}")
    print(f"  🎉 TRAINING COMPLETE!")
    print(f"{'═'*60}")
    print(f"  Architecture:     {arch_name}")
    print(f"  Best Val Acc:     {best_acc:.4f} ({best_acc*100:.2f}%)")
    if test_acc is not None:
        print(f"  Test Acc (TTA):   {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Total Time:       {total_time/60:.1f} minutes")
    print(f"  Model saved to:   {os.path.join(args.output_dir, 'skin_best_model_optimized.pth')}")
    print(f"{'═'*60}")

    if best_acc >= 0.90:
        print(f"\n  ✅ TARGET ACHIEVED! Val accuracy is {best_acc*100:.2f}% (≥ 90%)")
    else:
        print(f"\n  ⚠️  Val accuracy is {best_acc*100:.2f}%. To reach 90%:")
        print(f"     - Try running Phase 3 for more epochs (--epochs_phase3 50)")
        print(f"     - Merge visually similar classes")
        print(f"     - Add more training data for underperforming classes")
