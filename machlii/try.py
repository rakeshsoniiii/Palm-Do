# ✅ Install latest versions with onnxruntime
!pip install -q --upgrade timm albumentations opencv-python-headless torchmetrics insightface onnxruntime
!pip install numpy==2.0.0  # Compatible with thinc, opencv-python-headless, and potentially others

# === Imports ===
import os, json, time, math, random, glob
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import insightface

print("✅ All imports loaded successfully")
