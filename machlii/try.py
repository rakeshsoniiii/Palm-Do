# === Full Colab-ready Palmpay script with InsightFace integrated ===
# Paste into Google Colab. Run cells in order.

# 1) Install dependencies (run once in Colab)
# ✅ Recommended stable installation
!pip install -q timm albumentations opencv-python-headless torchmetrics insightface
# 2) Imports
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

# 3) Mount Google Drive (Colab)
from google.colab import drive
drive.mount('/content/drive')

# 4) Paths & config
DRIVE_ROOT = Path("/content/drive/MyDrive/palmpay")
CHECKPOINT_DIR = DRIVE_ROOT / "checkpoints"
EMBED_DIR = DRIVE_ROOT / "embeddings"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(EMBED_DIR, exist_ok=True)

# Place palms at: /content/drive/MyDrive/palmpay/palms/{p1,p2,...}/{img.jpg,...}
PALM_ROOT = DRIVE_ROOT / "palms"

# 5) Transforms
IMG_SIZE = 224
train_transform = A.Compose([
    A.RandomResizedCrop(IMG_SIZE, IMG_SIZE, scale=(0.9,1.0), ratio=(0.9,1.1)),
    A.Rotate(limit=25, p=0.8),
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.06, rotate_limit=10, p=0.6),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
    A.OneOf([A.GaussianBlur(3), A.MedianBlur(3), A.MotionBlur(3)], p=0.2),
    A.GaussNoise(var_limit=(5.0,30.0), p=0.2),
    A.CoarseDropout(max_holes=6, max_height=16, max_width=16, p=0.15),
    A.Normalize(),
    ToTensorV2()
])
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])

# 6) Dataset
class PalmDataset(Dataset):
    def _init_(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        persons = sorted([p.name for p in self.root_dir.iterdir() if p.is_dir()])
        self.person2idx = {p:i for i,p in enumerate(persons)}
        self.samples = []
        for p in persons:
            files = list((self.root_dir / p).glob("*"))
            for f in files:
                if f.suffix.lower() in [".jpg",".jpeg",".png"]:
                    self.samples.append((str(f), self.person2idx[p]))
    def _len_(self): return len(self.samples)
    def _getitem_(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)['image']
        return img, label

# 7) Embedding model and ArcFace loss (same as before)
class EmbeddingNet(nn.Module):
    def _init_(self, backbone_name="resnet18", emb_size=256, pretrained=True):
        super()._init_()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        in_feats = self.backbone.num_features
        self.fc = nn.Sequential(
            nn.Linear(in_feats, emb_size),
            nn.BatchNorm1d(emb_size),
        )
    def forward(self, x):
        f = self.backbone(x)
        e = self.fc(f)
        e = F.normalize(e, p=2, dim=1)
        return e

class ArcFaceLoss(nn.Module):
    def _init_(self, emb_size, num_classes, s=30.0, m=0.3, easy_margin=False):
        super()._init_()
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.randn(num_classes, emb_size))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.easy_margin = easy_margin
        self.ce = nn.CrossEntropyLoss()
    def forward(self, emb, labels):
        normed_weight = F.normalize(self.weight, p=2, dim=1)
        cos_t = F.linear(emb, normed_weight)
        cos_t = cos_t.clamp(-1.0+1e-7, 1.0-1e-7)
        target_cos = cos_t.gather(1, labels.view(-1,1)).view(-1)
        sin_t = torch.sqrt(1.0 - target_cos ** 2)
        cos_tm = target_cos * self.cos_m - sin_t * self.sin_m
        if self.easy_margin:
            cond = target_cos > 0
            final_target_cos = torch.where(cond, cos_tm, target_cos)
        else:
            cond_v = target_cos - self.th
            final_target_cos = torch.where(cond_v > 0, cos_tm, target_cos - self.mm)
        logits = cos_t.clone()
        logits.scatter_(1, labels.view(-1,1), final_target_cos.view(-1,1))
        logits = logits * self.s
        loss = self.ce(logits, labels)
        return loss, logits

# 8) Training helpers
def train_one_epoch(model, arcface, loader, opt, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in tqdm(loader, desc="train"):
        imgs = imgs.to(device)
        labels = labels.to(device)
        opt.zero_grad()
        emb = model(imgs)
        loss, _ = arcface(emb, labels)
        loss.backward()
        opt.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

def eval_embeddings(model, loader, device):
    model.eval()
    embeddings = []
    labels = []
    with torch.no_grad():
        for imgs, labs in tqdm(loader, desc="embed"):
            imgs = imgs.to(device)
            emb = model(imgs).cpu().numpy()
            embeddings.append(emb)
            labels.append(labs.numpy())
    embeddings = np.vstack(embeddings)
    labels = np.hstack(labels)
    return embeddings, labels

# 9) Load dataset (and split)
dataset = PalmDataset(PALM_ROOT, transform=train_transform)
num_classes = len(dataset.person2idx)
print("Persons found:", num_classes)
if num_classes < 2:
    raise SystemExit("Need at least 2 persons (folders) to fine-tune")

val_pct = 0.2
val_count = int(len(dataset) * val_pct)
train_count = len(dataset) - val_count
train_set, val_set = random_split(dataset, [train_count, val_count])
train_set.dataset.transform = train_transform
val_set.dataset.transform = val_transform

BATCH = 32
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

# 10) Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmbeddingNet(backbone_name="resnet18", emb_size=256, pretrained=True).to(device)
arcface = ArcFaceLoss(emb_size=256, num_classes=num_classes, s=30.0, m=0.35).to(device)
optimizer = torch.optim.AdamW(list(model.parameters()) + list(arcface.parameters()), lr=2e-4, weight_decay=1e-4)

# 11) Training loop
EPOCHS = 12
best_val_loss = 1e9
for epoch in range(1, EPOCHS+1):
    print(f"Epoch {epoch}/{EPOCHS} - {datetime.now().strftime('%H:%M:%S')}")
    train_loss = train_one_epoch(model, arcface, train_loader, optimizer, device)
    print("Train loss:", train_loss)
    emb_val, lab_val = eval_embeddings(model, val_loader, device)
    centroids = {}
    for c in np.unique(lab_val):
        centroids[c] = np.mean(emb_val[lab_val==c], axis=0)
    dists = []
    for i,l in enumerate(lab_val):
        q = emb_val[i]
        cent = centroids[l]
        cos_sim = np.dot(q,cent) / (np.linalg.norm(q)*np.linalg.norm(cent)+1e-8)
        dists.append(1 - cos_sim)
    val_loss = float(np.mean(dists))
    print("Val proxy loss (1-cos_sim):", val_loss)
    ckpt_path = CHECKPOINT_DIR / f"ckpt_epoch_{epoch}.pth"
    torch.save({
        'epoch':epoch,
        'model_state':model.state_dict(),
        'arcface_state':arcface.state_dict(),
        'opt_state':optimizer.state_dict(),
        'val_proxy_loss':val_loss
    }, str(ckpt_path))
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({'model_state':model.state_dict(),'arcface_state':arcface.state_dict()}, str(CHECKPOINT_DIR / "best_model.pth"))
    print("Best val proxy loss:", best_val_loss)

# 12) Export traced model
model.eval()
example = torch.randn(1,3,IMG_SIZE,IMG_SIZE).to(device)
traced = torch.jit.trace(model.cpu(), example.cpu())
traced.save(str(DRIVE_ROOT / "palmpay_model_traced.pt"))
print("Saved traced model.")

# 13) Save embeddings for convenience
train_full = PalmDataset(PALM_ROOT, transform=val_transform)
loader_full = DataLoader(train_full, batch_size=32, shuffle=False)
embs, labs = eval_embeddings(model, loader_full, device)
np.save(str(EMBED_DIR / "embeddings.npy"), embs)
np.save(str(EMBED_DIR / "labels.npy"), labs)
with open(str(EMBED_DIR / "person2idx.json"), "w") as f:
    json.dump(train_full.person2idx, f)
print("Saved embeddings and labels.")

# 14) InsightFace initialization (robust)
import insightface
from insightface.app import FaceAnalysis

# Choose ctx_id depending on GPU availability
ctx_id = 0 if torch.cuda.is_available() else -1
print("Initializing InsightFace (this may download models)... ctx_id =", ctx_id)
af_model = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider'])
# prepare will download and load models; ctx_id -1 uses CPU
try:
    af_model.prepare(ctx_id=ctx_id)
except Exception as e:
    # fallback: try CPU if GPU init fails
    print("InsightFace prepare failed with ctx_id", ctx_id, " — fallback to CPU. Error:", e)
    af_model = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    af_model.prepare(ctx_id=-1)

# 15) Embedding helpers (palm + face)
def image_to_palm_embedding(img_bgr, model, transform=val_transform, device=device):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = transform(image=img_rgb)['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(t).cpu().numpy().reshape(-1)
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb

def face_to_embedding(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = af_model.get(img_rgb)
    if len(faces) == 0:
        return None
    emb = faces[0].embedding  # insightface embedding (numpy), usually normalized
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb

# 16) Templates DB
TEMPLATE_DB_PATH = DRIVE_ROOT / "templates.json"
if TEMPLATE_DB_PATH.exists():
    with open(TEMPLATE_DB_PATH, "r") as f:
        templates = json.load(f)
else:
    templates = {}

def enroll_user(user_id, palm_imgs_bgr, face_imgs_bgr, model, device=device):
    palm_embs = [image_to_palm_embedding(img, model, transform=val_transform, device=device) for img in palm_imgs_bgr]
    palm_template = np.mean(palm_embs, axis=0)
    palm_template = palm_template / (np.linalg.norm(palm_template) + 1e-12)
    face_embs = []
    for img in face_imgs_bgr:
        fe = face_to_embedding(img)
        if fe is not None:
            face_embs.append(fe)
    face_template = None
    if face_embs:
        face_template = np.mean(face_embs, axis=0)
        face_template = face_template / (np.linalg.norm(face_template) + 1e-12)
    templates[user_id] = {
        "palm": palm_template.tolist(),
        "face": face_template.tolist() if face_template is not None else None,
        "name": user_id,
        "balance": 500,
        "created_at": datetime.now().isoformat()
    }
    with open(TEMPLATE_DB_PATH, "w") as f:
        json.dump(templates, f)
    print(f"Enrolled {user_id} with palm and face templates.")

# 17) Matching (fusion + online update)
def cosine(a,b):
    return float(np.dot(a,b) / ((np.linalg.norm(a)*np.linalg.norm(b))+1e-8))

def match_user(palm_img_bgr, face_img_bgr=None, model=model, palm_weight=0.7, face_weight=0.3, threshold=0.75):
    query_palm = image_to_palm_embedding(palm_img_bgr, model)
    query_face = face_to_embedding(face_img_bgr) if face_img_bgr is not None else None
    best = None
    best_score = -1.0
    for uid, rec in templates.items():
        rec_palm = np.array(rec["palm"])
        s_palm = cosine(query_palm, rec_palm)
        s_face = 0.0
        if query_face is not None and rec.get("face") is not None:
            rec_face = np.array(rec["face"])
            s_face = cosine(query_face, rec_face)
        total = palm_weight * s_palm + face_weight * s_face
        if total > best_score:
            best_score = total
            best = (uid, total, s_palm, s_face)
    if best_score >= threshold and best is not None:
        uid, total, s_palm, s_face = best
        # online template update for palm only (keeps face template stable)
        alpha = 0.92
        stored = np.array(templates[uid]["palm"])
        updated = alpha * stored + (1-alpha) * query_palm
        updated = updated / (np.linalg.norm(updated) + 1e-12)
        templates[uid]["palm"] = updated.tolist()
        with open(TEMPLATE_DB_PATH, "w") as f:
            json.dump(templates, f)
        return {"status":"match", "user_id":uid, "score":total, "palm_score":s_palm, "face_score":s_face, "balance": templates[uid]["balance"]}
    else:
        # quick demo enrol (uses single palm image if many not provided)
        new_id = f"user_{int(time.time())}"
        face_list = [face_img_bgr] if face_img_bgr is not None else []
        enroll_user(new_id, [palm_img_bgr], face_list, model)
        return {"status":"enrolled", "user_id":new_id, "balance": templates[new_id]["balance"]}

# 18) Simple liveness check (three-frame)
def simple_liveness_check(frames_bgr, threshold_pixels=20):
    prev = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
    alive = False
    for f in frames_bgr[1:]:
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        d = cv2.absdiff(prev, g)
        if d.mean() > threshold_pixels:
            alive = True
            break
        prev = g
    return alive

# 19) Webcam capture helpers
# 19a) Colab capture (JS) - Click "Take Photo" in the output to capture image
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
from google.colab import output

def _save_photo(data_url, filename):
    header, encoded = data_url.split(',', 1)
    data = b64decode(encoded)
    with open(filename, 'wb') as f:
        f.write(data)

def capture_image_colab(filename='capture.jpg'):
    js = Javascript('''
    async function takePhoto() {
      const div = document.createElement('div');
      const btn = document.createElement('button');
      btn.textContent = 'Take Photo';
      div.appendChild(btn);
      document.body.appendChild(div);
      const stream = await navigator.mediaDevices.getUserMedia({video: true});
      const video = document.createElement('video');
      video.style.width = '640px';
      video.style.height = '480px';
      video.srcObject = stream;
      await video.play();
      document.body.appendChild(video);
      await new Promise((resolve) => btn.onclick = resolve);
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getTracks().forEach(t => t.stop());
      const dataUrl = canvas.toDataURL('image/jpeg');
      google.colab.kernel.invokeFunction('save_photo', [dataUrl, %r], {});
    }
    takePhoto();
    ''' % filename)
    output.register_callback('save_photo', _save_photo)
    display(js)

# 19b) Local OpenCV capture (for laptop)
def capture_image_local(save_path='capture_local.jpg', show_preview=False):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Failed to capture frame")
    if show_preview:
        cv2.imshow("Preview - press any key to capture", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cv2.imwrite(save_path, frame)
    cap.release()
    return cv2.imread(save_path)

# 20) Usage examples (Colab demo)
print("Ready. Use capture_image_colab('face.jpg') or capture_image_local() to capture images.")
print("Then call enroll_user(user_id, [palm1,palm2,palm3], [face1], model) or match_user(palm, face, model).")

# Example quick usage (uncomment to run in Colab interactively)
# capture_image_colab('face.jpg')   # click Take Photo -> file saved
# img = cv2.imread('face.jpg')
# emb = face_to_embedding(img); print(emb.shape if emb is not None else "no face")
