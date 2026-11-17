# -*- coding: utf-8 -*-
"""
===============================================================================
 MODELE FINAL : DETECTION ET LOCALISATION (PROFONDEUR) DES VAISSEAUX
 U-Net ResNet34 (poids ImageNet), 20 époques, cible vessel + supervessel
===============================================================================

Script autonome : il ne garde que la configuration retenue pour la présentation
("imagenet_20ep" de comparaison_nuit.py), sans le pré-entraînement "ondes", les
autres configurations ni la machinerie de reprise multi-machines.

CE QUE FAIT LE MODELE
---------------------
Pour chaque élastogramme (image de 119 ou 129 lignes x 481 colonnes), il prédit :
  1. DETECTION    : la probabilité qu'un vaisseau soit présent (tête de classification) ;
  2. LOCALISATION : la profondeur du vaisseau. La carte du U-Net est moyennée sur
                    la largeur -> un score par ligne -> softmax = distribution de
                    probabilité sur les lignes ; la profondeur prédite est la médiane
                    de cette distribution autour de son pic (voir depth_from).

PROTOCOLE D'EVALUATION (figé avant les résultats, voir PROTOCOLE_CALIBRE.md)
---------------------------------------------------------------------------
 - 5 plis externes groupés par examen (GUID) : chaque image est testée une fois,
   par un modèle qui n'a jamais vu son examen.
 - Dans chaque train externe, ~1/6 des examens forment la CALIBRATION interne :
   ils n'entraînent pas le réseau et servent à fixer le seuil de présence par
   sonde (quantile 0,90 des scores des négatifs -> ~10 % de fausses alarmes).
 - Partitions lues dans sorties/final/partitions.json (identiques à celles de la
   comparaison des modèles) : pas de dépendance à la version de scikit-learn.
 - 20 époques fixes, pas de sélection d'époque ; 3 graines, puis ensemble
   (moyenne des prédictions des graines).
 - Référence pour la localisation : "ligne fixe" = médiane des profondeurs
   annotées de la même sonde dans le train, sans regarder l'image.
 - Intervalles de confiance à 95 % par bootstrap sur les examens.

UTILISATION
-----------
  python modele_final.py --evaluer       # quelques minutes : relit les prédictions
                                         #   sauvegardées et produit bilan + figures
  python modele_final.py --entrainer     # refait tout l'entraînement (3 graines x 5 plis,
                                         #   ~5 min/pli sur GPU, ~70 min/pli sur CPU),
                                         #   reprend au pli suivant si interrompu, puis évalue
  python modele_final.py --test-rapide   # 1 époque, 1 graine, 1 pli : vérifie l'installation

Entrées : Annotations/imgtag.annotations.json, Elastos/*.bmp et *.txt,
          sorties/final/partitions.json
Sorties (sorties/final/) :
  bilan_final.txt        bilan rédigé : détection, localisation, bout en bout
  resume_final.csv       chiffres principaux par graine et pour l'ensemble
  profondeur_final.png   profondeur prédite vs annotée (modèle et ligne fixe)
  courbes_final.png      loss d'entraînement / validation et learning rate
  seuils_final.json      seuils de présence calibrés par pli et par sonde
  resultats_final.csv    prédictions image par image (contient les noms d'images
                         et les GUID d'examens : NE PAS diffuser)
  predictions/           prédictions de chaque graine x pli (.npz) et historiques
                         de loss (.history.csv) (contiennent les noms d'images)
  journal_*.txt          tout ce qui a été affiché
"""

import os
# DOIT être défini AVANT "import torch" (voir pipeline_vaisseaux.py) :
# masque le GPU Intel (plantage du pilote) et autorise le repli CPU sur Mac.
os.environ.setdefault("ONEAPI_DEVICE_SELECTOR", "opencl:cpu")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score

import segmentation_models_pytorch as smp

# =============================================================================
# CONFIGURATION DU MODELE RETENU (ne pas modifier : ce sont les réglages évalués)
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
JSON_PATH = BASE_DIR / "Annotations" / "imgtag.annotations.json"
IMG_DIR = BASE_DIR / "Elastos"                  # images .bmp et métadonnées .txt
FINAL_DIR = BASE_DIR / "sorties" / "final"
PARTITIONS = FINAL_DIR / "partitions.json"
PRED_DIR = FINAL_DIR / "predictions"

VESSEL_LABELS = ("vessel", "supervessel")       # cible = union des deux labels
ENCODER = "resnet34"                            # encodeur, poids ImageNet
EPOCHS = 20                                     # nombre d'époques fixe
BS = 4                                          # taille de lot
LR_ENC = 1e-4                                   # learning rate encodeur (fine-tuning doux)
LR_DEC = 1e-3                                   # learning rate décodeur + têtes
LAMBDA = 1.0                                    # poids de la perte de profondeur
SEEDS = [0, 1, 2]                               # graines du modèle ; ensemble = moyenne
DECODER_CHANNELS = (256, 128, 64, 32, 16)       # décodeur standard de smp

SPEC = 0.90                                     # spécificité visée pour le seuil de présence
FRAC = 0.5                                      # bande prédite : lignes >= FRAC x pic
TOLS = (0, 5)                                   # tolérances (px) autour de la bande annotée
N_BOOT = 2000                                   # rééchantillonnages bootstrap
DIAG_EVERY = 5                                  # diagnostic sur la calibration toutes les N époques

CROP_W = 320    # largeur du recadrage aléatoire à l'entraînement (sur 481 px)
HP = 160        # hauteur de stockage des distributions de profondeur (>= 129)
WP = 512        # largeur après padding en évaluation (multiple de 32, >= 481)

# Vitesse uniquement (aucun effet sur les résultats)
N_THREADS = 10                                  # threads CPU ; None = choix de PyTorch
CHANNELS_LAST = True                            # format mémoire plus rapide (sauf Mac)
# =============================================================================

if torch.cuda.is_available():
    DEV = "cuda"
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    DEV = "mps"
else:
    DEV = "cpu"
if DEV == "cpu" and N_THREADS:
    torch.set_num_threads(N_THREADS)
if DEV == "cuda":
    torch.backends.cudnn.benchmark = True
MEM_FMT = torch.channels_last if CHANNELS_LAST and DEV != "mps" else torch.contiguous_format


def to_dev(x):
    return x.to(DEV, non_blocking=(DEV == "cuda")).contiguous(memory_format=MEM_FMT)


def pad_height(h):
    """Plus petit multiple de 32 >= h (exigé par le U-Net) : 119 -> 128, 129 -> 160."""
    return -(-h // 32) * 32


class Tee:
    """Écrit à la fois dans la console et dans le journal."""

    def __init__(self, path, stream):
        self.f = open(path, "a", encoding="utf-8")
        self.s = stream

    def write(self, text):
        self.s.write(text)
        self.f.write(text)
        self.f.flush()

    def flush(self):
        self.s.flush()
        self.f.flush()


# =============================================================================
# 1. DONNEES
# =============================================================================
def build_table():
    """Table des images exploitables : (name, guid, H, probe, lab, strat).
    lab = 1 si vessel OU supervessel est présent (tag ET boîte cohérents),
    0 si les deux sont absents, image exclue sinon (label incertain, "weird",
    "double:yes")."""
    data = json.load(open(JSON_PATH, encoding="utf-8"))["images"]
    rows = []
    for name, e in data.items():
        stem = name[:-4]
        tags = set(e["tags"])
        probe = None
        for line in (IMG_DIR / (stem + ".txt")).read_text(errors="replace").splitlines():
            if line.upper().startswith("PROBE MODEL"):
                probe = int(float(line.split("=")[1].split()[0]))
        guid = stem.split("_")[2]                      # identifiant de l'examen
        states = []
        for k in VESSEL_LABELS:
            n_boxes = sum(b["label"] == k for b in e["boxes"])
            if f"{k}:yes" in tags and n_boxes > 0:
                states.append("oui")
            elif f"{k}:no" in tags and n_boxes == 0:
                states.append("non")
            elif f"{k}:yes" not in tags and f"{k}:no" not in tags and n_boxes == 0:
                states.append("inconnu")
            else:
                states.append("incohérent")
        if "incohérent" in states:
            lab = -1
        elif "oui" in states:
            lab = 1
        elif all(s == "non" for s in states):
            lab = 0
        else:
            lab = -1
        rows.append(dict(name=name, guid=guid, H=e["height"], probe=probe, lab=lab,
                         weird="weird" in tags, double="double:yes" in tags))
    df = pd.DataFrame(rows)
    L = df[(df.lab >= 0) & ~df.weird & ~df.double].reset_index(drop=True)
    L["strat"] = L.probe * 2 + L.lab
    print(f"{len(df)} images annotées -> {len(L)} exploitables ({int(L.lab.sum())} avec vaisseau), "
          f"{L.guid.nunique()} examens")
    return data, L


def build_cache(data, L):
    """Images normalisées (moyenne 0, écart-type 1 par image) et cibles :
      iv  : intervalles de lignes annotés (union des boîtes vessel + supervessel)
      t   : cible de profondeur = distribution uniforme sur ces lignes
      med : ligne médiane annotée (NaN pour une image sans vaisseau)"""
    cache = {}
    for _, r in L.iterrows():
        a = np.asarray(Image.open(IMG_DIR / r["name"]).convert("L"), np.float32)
        a = (a - a.mean()) / (a.std() + 1e-6)
        m = np.zeros(r.H, np.float32)
        for b in data[r["name"]]["boxes"]:
            if b["label"] in VESSEL_LABELS:
                m[b["y"]:min(b["y"] + b["h"] - 1, r.H - 1) + 1] = 1
        iv = []
        for y in np.flatnonzero(m):
            if iv and y == iv[-1][1] + 1:
                iv[-1] = (iv[-1][0], int(y))
            else:
                iv.append((int(y), int(y)))
        t = m / m.sum() if m.sum() > 0 else m
        med = float(np.median(np.where(m > 0)[0])) if m.sum() > 0 else np.nan
        cache[r["name"]] = dict(img=a, iv=iv, t=t, med=med)
    return cache


def load_splits(table):
    """Partitions train / calibration / test figées, avec contrôles : mêmes images,
    chaque image testée une fois, aucun examen partagé entre train, calibration et
    test, deux classes par sonde dans train et calibration."""
    saved = json.loads(PARTITIONS.read_text(encoding="utf-8"))
    if saved["names"] != table.name.tolist():
        raise ValueError("partitions.json ne correspond pas aux images actuelles (annotations modifiées ?)")
    splits = saved["splits"]
    coverage = np.zeros(len(table), int)
    for s in splits:
        parts = [np.asarray(s[k], dtype=int) for k in ("train", "cal", "test")]
        if not np.array_equal(np.sort(np.concatenate(parts)), np.arange(len(table))):
            raise ValueError("Partitions incomplètes ou indices dupliqués")
        g = [set(table.guid.iloc[p]) for p in parts]
        if g[0] & g[1] or g[0] & g[2] or g[1] & g[2]:
            raise ValueError("Un examen apparaît dans deux parties d'un même pli")
        for k in ("train", "cal"):
            sub = table.iloc[s[k]]
            for pr in (7, 9):
                if set(sub.loc[sub.probe == pr, "lab"]) != {0, 1}:
                    raise ValueError(f"{k} sans les deux classes pour la sonde {pr}")
        coverage[parts[2]] += 1
    if not np.all(coverage == 1):
        raise ValueError("Chaque image doit être testée exactement une fois")
    print(f"Partitions vérifiées : {len(splits)} plis, aucun examen partagé entre train, calibration et test")
    return splits


class DS(torch.utils.data.Dataset):
    """Entraînement : recadrage aléatoire en largeur (la profondeur ne bouge pas) et
    léger changement de contraste. Évaluation : image entière. Padding en bas et à
    droite : les lignes gardent leur numéro (= la profondeur)."""

    def __init__(self, d, cache, train):
        self.d, self.c, self.train = d.reset_index(drop=True), cache, train

    def __len__(self):
        return len(self.d)

    def __getitem__(self, i):
        r = self.d.iloc[i]
        e = self.c[r["name"]]
        a = e["img"]
        H, W = a.shape
        if self.train:
            s = np.random.randint(0, W - CROP_W + 1)
            a = a[:, s:s + CROP_W]
            a = a * np.random.uniform(.9, 1.1) + np.random.uniform(-.1, .1)
        w = a.shape[1]
        hp = pad_height(H)
        x = np.zeros((1, hp, CROP_W if self.train else WP), np.float32)
        x[0, :H, :w] = a
        t = np.zeros(hp, np.float32)
        t[:H] = e["t"]
        v = np.zeros(hp, np.float32)
        v[:H] = 1                                      # lignes réelles (hors padding)
        return torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(v), np.float32(r.lab), w


class HeightBatchSampler(torch.utils.data.Sampler):
    """Lots d'images de même hauteur paddée (une seule sonde par lot)."""

    def __init__(self, heights, bs, base=None):
        self.hp = np.array([pad_height(h) for h in heights])
        self.bs, self.base = bs, base

    def __iter__(self):
        if self.base is None:                          # évaluation : tout, dans l'ordre
            for h in np.unique(self.hp):
                idx = np.where(self.hp == h)[0].tolist()
                for k in range(0, len(idx), self.bs):
                    yield idx[k:k + self.bs]
            return
        buckets = {}
        for i in self.base:                            # entraînement : ordre aléatoire
            b = buckets.setdefault(self.hp[i], [])
            b.append(int(i))
            if len(b) == self.bs:
                yield b
                buckets[self.hp[i]] = []

    def __len__(self):
        if self.base is None:
            return sum(-(-int((self.hp == h).sum()) // self.bs) for h in np.unique(self.hp))
        return len(self.base) // self.bs


def make_balanced_sampler(tr):
    """Tirage équilibré 50/50 entre classes À L'INTÉRIEUR de chaque sonde (le
    déséquilibre est de sens opposé selon la sonde), sur le train uniquement."""
    tr = tr.reset_index(drop=True)
    n_probe = tr.groupby("probe").size()
    n_group = tr.groupby(["probe", "lab"]).size()
    w = [n_probe[p] / 2 / n_group[(p, l)] for p, l in zip(tr.probe, tr.lab)]
    return torch.utils.data.WeightedRandomSampler(torch.tensor(w, dtype=torch.double),
                                                  num_samples=len(tr), replacement=True)


# =============================================================================
# 2. MODELE, PERTE, PREDICTION
# =============================================================================
class Net(nn.Module):
    """U-Net + tête de classification. rows : un logit par ligne (carte moyennée
    sur la largeur utile) ; lab : logit de présence."""

    def __init__(self, encoder_weights="imagenet"):
        super().__init__()
        self.m = smp.Unet(ENCODER, encoder_weights=encoder_weights, in_channels=1, classes=1,
                          decoder_channels=DECODER_CHANNELS, aux_params=dict(classes=1, dropout=0.2))

    def forward(self, x, w):
        seg, lab = self.m(x)
        return seg[..., :w].mean(-1).squeeze(1), lab.squeeze(1)


def loss_fn(rows, lab, t, v, y):
    """BCE de présence (toutes les images) + LAMBDA x entropie croisée entre la cible
    de profondeur et le softmax des lignes (images avec vaisseau seulement)."""
    bce = F.binary_cross_entropy_with_logits(lab, y)
    logq = F.log_softmax(rows.masked_fill(v == 0, -1e4), dim=1)
    pos = y > .5
    depth_loss = -(t * logq).sum(1)[pos].mean() if pos.any() else rows.sum() * 0
    return bce + LAMBDA * depth_loss


def predict(model, d, cache):
    """P : probabilité de présence (N,) ; Q : distribution sur les lignes (N, HP)."""
    model.eval()
    d = d.reset_index(drop=True)
    P = np.zeros(len(d), np.float32)
    Q = np.zeros((len(d), HP), np.float32)
    batches = list(HeightBatchSampler(d.H.values, BS))
    dl = torch.utils.data.DataLoader(DS(d, cache, False), batch_sampler=batches)
    with torch.inference_mode():
        for idx, (x, t, v, y, w) in zip(batches, dl):
            rows, lab = model(to_dev(x), int(w[0]))
            if not torch.isfinite(rows).all().item() or not torch.isfinite(lab).all().item():
                raise FloatingPointError("Prédiction NaN/Inf")
            q = F.softmax(rows.float().cpu().masked_fill(v == 0, -1e4), 1).numpy()
            P[idx] = torch.sigmoid(lab).float().cpu().numpy()
            Q[idx, :q.shape[1]] = q
    return P, Q


def check_predictions(p, q, n):
    if p.shape != (n,) or q.shape != (n, HP):
        raise ValueError("Dimensions des prédictions incompatibles")
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise FloatingPointError("Prédictions NaN/Inf")
    if np.any((p < 0) | (p > 1)) or np.any(q < 0) or not np.allclose(q.sum(1), 1, atol=1e-5):
        raise ValueError("Probabilités invalides")


# =============================================================================
# 3. PROFONDEUR ET STATISTIQUES
# =============================================================================
def depth_from(q):
    """Profondeur prédite : pic de q, extension de la bande contiguë tant que
    q >= FRAC x pic, puis médiane de la probabilité sur cette bande.
    Retourne (profondeur, haut de la bande, bas de la bande)."""
    k = int(q.argmax())
    lo = hi = k
    while lo > 0 and q[lo - 1] >= FRAC * q[k]:
        lo -= 1
    while hi < len(q) - 1 and q[hi + 1] >= FRAC * q[k]:
        hi += 1
    cum = np.cumsum(q[lo:hi + 1])
    return lo + int(np.searchsorted(cum, cum[-1] / 2)), lo, hi


def inside(iv, d, tol):
    """d tombe dans au moins un intervalle annoté (élargi de tol px)."""
    return any(a - tol <= d <= b + tol for a, b in iv)


def dist_to_interval(iv, d):
    """Distance (px) à l'intervalle annoté le plus proche (0 si dedans)."""
    return min(max(0, a - d, d - b) for a, b in iv)


def boot_auc(y, s, g, n=N_BOOT // 2):
    """IC95 de l'AUC, bootstrap sur les examens (GUID)."""
    ug = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in ug}
    rng = np.random.default_rng(0)
    res = []
    for _ in range(n):
        ii = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
        if len(set(y[ii])) == 2:
            res.append(roc_auc_score(y[ii], s[ii]))
    return np.percentile(res, [2.5, 97.5]) if res else (np.nan, np.nan)


def boot_diff(a, b, g, n=N_BOOT):
    """IC95 de la différence de deux taux appariés (a - b), bootstrap sur les examens."""
    ug, inv = np.unique(g, return_inverse=True)
    sa = np.bincount(inv, a.astype(float))
    sb = np.bincount(inv, b.astype(float))
    cn = np.bincount(inv, np.ones(len(a)))
    rng = np.random.default_rng(0)
    res = []
    for _ in range(n):
        s = rng.integers(0, len(ug), len(ug))
        res.append((sa[s].sum() - sb[s].sum()) / cn[s].sum())
    return np.percentile(res, [2.5, 97.5])


def detection_counts(present, y):
    """(sensibilité, spécificité, VPP, VPN, exactitude, VP, FN, VN, FP)."""
    vp = int((present & (y == 1)).sum()); fn = int((~present & (y == 1)).sum())
    vn = int((~present & (y == 0)).sum()); fp = int((present & (y == 0)).sum())
    div = lambda a, b: a / b if b else np.nan
    return (div(vp, vp + fn), div(vn, vn + fp), div(vp, vp + fp), div(vn, vn + fn),
            div(vp + vn, len(y)), vp, fn, vn, fp)


# =============================================================================
# 4. ENTRAINEMENT
# =============================================================================
@torch.no_grad()
def validation_loss(model, data, cache):
    """Loss sur la calibration (diagnostic) : BCE moyenne + LAMBDA x profondeur moyenne."""
    previous_mode = model.training
    model.eval()
    data = data.reset_index(drop=True)
    loader = torch.utils.data.DataLoader(
        DS(data, cache, False), batch_sampler=HeightBatchSampler(data.H.values, BS),
        generator=torch.Generator().manual_seed(0))    # ne consomme pas l'aléa d'entraînement
    bce_sum = depth_sum = 0.0
    count = positives = 0
    try:
        for x, t, v, y, w in loader:
            rows, lab = model(to_dev(x), int(w[0]))
            rows, lab = rows.float(), lab.float()
            t, v, y = t.to(DEV), v.to(DEV), y.to(DEV)
            bce_sum += F.binary_cross_entropy_with_logits(lab, y, reduction="sum").item()
            logq = F.log_softmax(rows.masked_fill(v == 0, -1e4), dim=1)
            pos = y > .5
            depth_sum += (-(t[pos] * logq[pos]).sum()).item()
            count += len(y)
            positives += int(pos.sum().item())
        bce_mean = bce_sum / count
        depth_mean = depth_sum / positives if positives else 0.0
        if not np.isfinite(bce_mean) or not np.isfinite(depth_mean):
            raise FloatingPointError("Loss de validation NaN/Inf")
        return dict(val_loss=bce_mean + LAMBDA * depth_mean, val_presence_loss=bce_mean,
                    val_depth_loss=depth_mean, val_images=count, val_positives=positives)
    finally:
        model.train(previous_mode)


def train_fold(tr, ca, cache, seed, history_path):
    """Entraîne un modèle EPOCHS époques sur `tr` (modèle de la dernière époque).
    `ca` (calibration) ne sert qu'aux diagnostics. Retourne (modèle, learning rates).

    Reproductibilité : l'ordre des opérations aléatoires est celui de
    pipeline_vaisseaux.train_fold (graine, création du réseau, tirage équilibré,
    diagnostic toutes les DIAG_EVERY époques : chaque DataLoader consomme un tirage
    du générateur de torch). Ne pas réordonner sans accepter d'autres résultats."""
    tr = tr.reset_index(drop=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = Net().to(DEV, memory_format=MEM_FMT)
    opt = torch.optim.AdamW([
        {"params": model.m.encoder.parameters(), "lr": LR_ENC},
        {"params": list(model.m.decoder.parameters()) + list(model.m.segmentation_head.parameters())
                   + list(model.m.classification_head.parameters()), "lr": LR_DEC},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)   # décroissance en cosinus
    dl = torch.utils.data.DataLoader(
        DS(tr, cache, True), batch_sampler=HeightBatchSampler(tr.H.values, BS, make_balanced_sampler(tr)))
    history, lrs, curve = [], [], []
    for ep in range(EPOCHS):
        t0 = time.time()
        model.train()
        total, n_steps = 0.0, 0
        for x, t, v, y, w in dl:
            x = to_dev(x)
            t, v, y = t.to(DEV), v.to(DEV), y.to(DEV)
            rows, lab = model(x, int(w[0]))
            loss = loss_fn(rows, lab, t, v, y)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Perte NaN/Inf à l'époque {ep + 1}, lot {n_steps + 1}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.detach()
            n_steps += 1
        lrs.append([g["lr"] for g in opt.param_groups])  # learning rate utilisé à cette époque
        sched.step()
        total = float(total) / max(n_steps, 1)
        val = validation_loss(model, ca, cache)
        history.append(dict(epoch=ep + 1, train_loss_batch_mean=total, **val))
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(f"    époque {ep + 1}/{EPOCHS} ({time.time() - t0:.0f} s) | loss train {total:.4f} | "
              f"loss validation {val['val_loss']:.4f}", flush=True)
        if DIAG_EVERY and (ep % DIAG_EVERY == DIAG_EVERY - 1 or ep == EPOCHS - 1):
            P, Q = predict(model, ca, cache)
            vv = ca.reset_index(drop=True)
            ins = [inside(cache[vv.name[i]]["iv"], depth_from(Q[i][:vv.H[i]])[0], 0)
                   for i in np.where(vv.lab == 1)[0]]
            auc = roc_auc_score(vv.lab, P) if vv.lab.nunique() == 2 else float("nan")
            curve.append(f"ep{ep + 1}: AUC calibration {auc:.3f} | profondeur dans la bande {np.mean(ins):.2f}")
    print("    " + "\n    ".join(curve))
    return model, np.array(lrs, dtype=float)


def train_all(table, cache, splits, seeds, folds, out_dir):
    """Entraîne graines x plis ; un pli déjà calculé (fichier .npz) est sauté."""
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [(s, f) for s in seeds for f in folds if not (out_dir / f"seed{s}_fold{f}.npz").exists()]
    print(f"\nEntraînement sur {DEV} : {len(todo)} pli(s) à calculer sur {len(seeds) * len(folds)}", flush=True)
    for k, (seed, f) in enumerate(todo, 1):
        tr, ca, te = [table.iloc[splits[f][key]] for key in ("train", "cal", "test")]
        stem = out_dir / f"seed{seed}_fold{f}"
        print(f"\n[{k}/{len(todo)}] graine {seed}, pli {f + 1} | train {len(tr)}, "
              f"calibration {len(ca)}, test {len(te)}", flush=True)
        t0 = time.time()
        model, lr = train_fold(tr, ca, cache, seed, stem.with_suffix(".history.csv"))
        pc, qc = predict(model, ca, cache); check_predictions(pc, qc, len(ca))
        pt, qt = predict(model, te, cache); check_predictions(pt, qt, len(te))
        tmp = stem.with_suffix(".tmp.npz")
        np.savez(tmp, p=pt, q=qt, cal_p=pc, cal_q=qc, lr=lr,
                 test_names=te.name.to_numpy(dtype=str), cal_names=ca.name.to_numpy(dtype=str))
        os.replace(tmp, stem.with_suffix(".npz"))      # écriture atomique : pas de fichier à moitié écrit
        del model
        if DEV == "mps":
            torch.mps.empty_cache()
        print(f"  pli terminé en {(time.time() - t0) / 60:.1f} min", flush=True)


# =============================================================================
# 5. EVALUATION
# =============================================================================
def load_predictions(table, splits, seeds, folds, pred_dir):
    """Rassemble les prédictions de test (et de calibration) de chaque graine."""
    runs = []
    for s in seeds:
        p, q = np.zeros(len(table)), np.zeros((len(table), HP), np.float32)
        cal, hist, lr = [], [], None
        for f in folds:
            sp = splits[f]
            with np.load(pred_dir / f"seed{s}_fold{f}.npz") as z:
                if (z["test_names"].tolist() != table.name.iloc[sp["test"]].tolist()
                        or z["cal_names"].tolist() != table.name.iloc[sp["cal"]].tolist()):
                    raise ValueError(f"seed{s}_fold{f}.npz : images différentes des partitions")
                check_predictions(z["p"], z["q"], len(sp["test"]))
                p[sp["test"]], q[sp["test"]] = z["p"], z["q"]
                cal.append(z["cal_p"])
                lr = z["lr"] if lr is None else lr
            h = pred_dir / f"seed{s}_fold{f}.history.csv"
            if h.exists():
                hist.append(pd.read_csv(h))
        runs.append(dict(seed=s, p=p, q=q, cal=cal, hist=hist, lr=lr))
    return runs


def calibrate(scores, ca):
    """Seuil par sonde = quantile SPEC des scores des négatifs de calibration."""
    out = {}
    for pr in (7, 9):
        neg = scores[(ca.probe.to_numpy() == pr) & (ca.lab.to_numpy() == 0)]
        if not len(neg):
            raise ValueError(f"Pas de négatifs de calibration pour la sonde {pr}")
        out[pr] = float(np.quantile(neg, SPEC))
    return out


def evaluate(name, p, Q, cal, table, splits, cache, out_dir=None):
    """Évalue des prédictions hors-échantillon. Retourne (ligne de résumé, détails).
    Si out_dir est donné, écrit aussi seuils, CSV par image et figure de profondeur."""
    N = len(table)
    y, probe, H = table.lab.values, table.probe.values, table.H.values
    guid = np.array(table.guid.tolist(), dtype=object)
    pos = np.where(y == 1)[0]
    med = np.array([cache[n]["med"] for n in table.name])
    iv = [cache[n]["iv"] for n in table.name]

    # Seuils de présence : calibrés sur la calibration interne de chaque pli,
    # appliqués aux images de test de ce pli.
    thr = np.zeros(N)
    seuils = []
    for sp, pc in zip(splits, cal):
        ca, te = table.iloc[sp["cal"]], table.iloc[sp["test"]]
        v = calibrate(pc, ca)
        seuils.append(dict(seuils={str(k): x for k, x in v.items()},
                           negatifs_calibration={str(pr): int(((ca.probe == pr) & (ca.lab == 0)).sum())
                                                 for pr in (7, 9)}))
        thr[sp["test"]] = [v[int(pr)] for pr in te.probe]
    present = p > thr

    # Profondeur prédite et baseline "ligne fixe" (médiane du train de la même sonde).
    dd = [depth_from(Q[i][:H[i]]) for i in range(N)]
    dep = np.array([a for a, _, _ in dd])
    base = np.zeros(N)
    for sp in splits:
        tr, te = np.array(sp["train"]), np.array(sp["test"])
        for pr in (7, 9):
            base[te[probe[te] == pr]] = np.median([med[i] for i in tr if y[i] == 1 and probe[i] == pr])

    out, det = {"run": name}, {"n": N, "n_pos": len(pos)}
    print(f"\n{'=' * 14} {name} {'=' * 14}")

    # Détection : AUC par sonde (indépendante du seuil)
    for pr in (7, 9):
        m = probe == pr
        auc = roc_auc_score(y[m], p[m])
        lo, hi = boot_auc(y[m], p[m], guid[m])
        det[f"auc sonde {pr}"] = (auc, lo, hi, int(m.sum()), int(y[m].sum()))
        out[f"AUC s{pr}"] = auc
        print(f"  AUC sonde {pr} : {auc:.3f} [IC95 {lo:.3f}-{hi:.3f}]")

    # Localisation : sur toutes les images avec vaisseau
    err_m, err_b = np.abs(dep[pos] - med[pos]), np.abs(base[pos] - med[pos])
    dist_m = np.array([dist_to_interval(iv[i], dep[i]) for i in pos])
    dist_b = np.array([dist_to_interval(iv[i], base[i]) for i in pos])
    ins = {}
    for tol in TOLS:
        im = np.array([inside(iv[i], dep[i], tol) for i in pos])
        ib = np.array([inside(iv[i], base[i], tol) for i in pos])
        ins[tol] = im
        lo, hi = boot_diff(im, ib, guid[pos])
        for pr in (7, 9):
            mm = probe[pos] == pr
            lo2, hi2 = boot_diff(im[mm], ib[mm], guid[pos][mm])
            det[f"loc tol{tol} sonde {pr}"] = (im[mm].mean(), ib[mm].mean(), lo2, hi2, int(mm.sum()),
                                               np.median(dist_m[mm]), np.median(err_m[mm]))
            if tol == 0:
                out[f"dedans s{pr}"], out[f"baseline s{pr}"] = im[mm].mean(), ib[mm].mean()
        det[f"loc tol{tol} global"] = (im.mean(), ib.mean(), lo, hi, len(pos),
                                       np.median(dist_m), np.median(err_m))
        print(f"  dans la bande (tol {tol} px) : modèle {im.mean():.1%} | ligne fixe {ib.mean():.1%} "
              f"| écart IC95 [{lo:+.1%} ; {hi:+.1%}]")
        if tol == 0:
            out["dedans global"], out["baseline global"] = im.mean(), ib.mean()
            out["diff IC95"] = f"[{lo:+.1%} ; {hi:+.1%}]"
            out["err méd px"] = np.median(err_m)
    det["err base"] = (np.median(err_b), np.median(dist_b))
    det["dist p90"] = (np.percentile(dist_m, 90), np.percentile(dist_b, 90))

    # Décision avec seuil + bout en bout (détecté ET dans la bande)
    for pr in (7, 9):
        m = probe == pr
        sel = probe[pos] == pr
        e0 = (present[pos][sel] & ins[0][sel]).mean()
        e5 = (present[pos][sel] & ins[TOLS[1]][sel]).mean()
        det[f"seuil sonde {pr}"] = detection_counts(present[m], y[m]) + (e0, e5)
        out[f"sens s{pr}"], out[f"bout-en-bout s{pr}"] = det[f"seuil sonde {pr}"][0], e0
        print(f"  sonde {pr} : sensibilité {out[f'sens s{pr}']:.3f} | spécificité "
              f"{det[f'seuil sonde {pr}'][1]:.3f} | bout en bout {e0:.3f}")
    det["seuil global"] = detection_counts(present, y) + (
        (present[pos] & ins[0]).mean(), (present[pos] & ins[TOLS[1]]).mean())

    if out_dir is not None:
        (out_dir / "seuils_final.json").write_text(json.dumps(seuils, indent=2), encoding="utf-8")
        res = table[["name", "guid", "probe", "lab"]].copy()
        res["p_present"], res["seuil"], res["present"] = p, thr, present
        res["depth_pred"] = dep
        res["band_lo"], res["band_hi"] = [b for _, b, _ in dd], [c for _, _, c in dd]
        res["gt_median"] = med
        res["gt_intervals"] = [str(v) if len(v) else "" for v in iv]
        res["inside_tol0"] = [inside(iv[i], dep[i], 0) if y[i] == 1 else np.nan for i in range(N)]
        res["dist_to_interval"] = [dist_to_interval(iv[i], dep[i]) if y[i] == 1 else np.nan for i in range(N)]
        res.to_csv(out_dir / "resultats_final.csv", index=False)
        plot_depth(out_dir / "profondeur_final.png", med, dep, base, probe, pos)
    return out, det


# =============================================================================
# 6. BILAN ET FIGURES
# =============================================================================
def write_bilan(d, S, n_seeds, path):
    """Bilan rédigé du modèle final (ensemble des graines)."""
    L = []
    w = L.append
    pct = lambda x: f"{100 * x:.1f} %"

    def seeds_line(what, c7, c9):
        if n_seeds < 2:
            return
        f = lambda c: (f"{100 * S[c].iloc[:n_seeds].astype(float).mean():.1f} ± "
                       f"{100 * S[c].iloc[:n_seeds].astype(float).std():.1f} %")
        w(f"   Variabilité entre les {n_seeds} graines seules ({what}) : sonde 7 {f(c7)} | sonde 9 {f(c9)}")

    w("=" * 78)
    w(f"BILAN DU MODELE FINAL : U-Net {ENCODER} (ImageNet), {EPOCHS} époques")
    w("=" * 78)
    w(f"{d['n']} images évaluées, dont {d['n_pos']} avec vaisseau (cible : union de "
      f"{' + '.join(VESSEL_LABELS)}).")
    w("Chaque image est prédite par un modèle qui n'a jamais vu son examen : ces chiffres")
    w("estiment la performance sur des examens non vus (patients distincts non vérifiés).")
    w("[IC95] = intervalle de confiance à 95 % (bootstrap par examen).")
    if n_seeds > 1:
        w(f"Modèle évalué : ENSEMBLE des {n_seeds} graines (moyenne de leurs prédictions).")

    w("")
    w("1. DETECTION : le modèle trouve-t-il les vaisseaux ?")
    w("-" * 78)
    w("a) Qualité du classement (AUC, indépendante de tout seuil). 0,5 = hasard, 1 = parfait.")
    w("   Lecture : probabilité qu'une image avec vaisseau reçoive un score plus élevé")
    w("   qu'une image sans vaisseau.")
    for pr in (7, 9):
        auc, lo, hi, n, npos = d[f"auc sonde {pr}"]
        w(f"   sonde {pr} : AUC {auc:.3f} [IC95 {lo:.3f} - {hi:.3f}]   ({n} images, {npos} avec vaisseau)")
    w("   (pas d'AUC globale : elle serait gonflée par la différence de proportion de")
    w("   vaisseaux entre les deux sondes)")
    w("")
    w(f"b) Décision présent / absent, seuil fixé par sonde pour {pct(1 - SPEC)} de fausses alarmes visées")
    w("   (seuil calculé sur la calibration interne, sans les images testées).")
    for key, lab in (("seuil sonde 7", "sonde 7"), ("seuil sonde 9", "sonde 9"), ("seuil global", "ensemble")):
        se, sp, vpp, vpn, acc, vp, fn, vn, fp = d[key][:9]
        w(f"   {lab:8s}: sensibilité {pct(se)} ({vp}/{vp + fn} vaisseaux trouvés) | "
          f"spécificité {pct(sp)} ({vn}/{vn + fp})")
        w(f"             VPP {pct(vpp)} (quand il dit 'vaisseau', il a raison) | "
          f"VPN {pct(vpn)} | exactitude {pct(acc)}")
    seeds_line("sensibilité", "sens s7", "sens s9")

    w("")
    w("2. LOCALISATION : sur les images AVEC vaisseau, le modèle le place-t-il au bon endroit ?")
    w("-" * 78)
    w("   Évaluée sur toutes les images avec vaisseau, que le modèle l'ait détecté ou non :")
    w("   on mesure la localisation seule. Référence = 'ligne fixe' (même profondeur pour")
    w("   toutes les images d'une sonde, médiane du train, sans regarder l'image).")
    w("   Le modèle n'apporte quelque chose que s'il fait mieux.")
    for tol in TOLS:
        w(f"   Profondeur prédite DANS la bande annotée (tolérance {tol} px) :")
        for k, lab in ((f"loc tol{tol} sonde 7", "sonde 7"), (f"loc tol{tol} sonde 9", "sonde 9"),
                       (f"loc tol{tol} global", "ensemble")):
            m_, b_, lo, hi, n = d[k][:5]
            verdict = ("MIEUX que la ligne fixe" if lo > 0 else "MOINS BIEN que la ligne fixe" if hi < 0
                       else "pas de différence démontrée avec la ligne fixe")
            w(f"     {lab:8s}: modèle {pct(m_)} | ligne fixe {pct(b_)} | écart {100 * (m_ - b_):+.1f} pts "
              f"[IC95 {100 * lo:+.1f} ; {100 * hi:+.1f}] -> {verdict} (n={n})")
    seeds_line("dans la bande, tolérance 0", "dedans s7", "dedans s9")
    dm, em = d["loc tol0 global"][5:7]
    eb, db = d["err base"]
    p90m, p90b = d["dist p90"]
    w(f"   Erreur typique : distance médiane à la bande {dm:.1f} px (ligne fixe {db:.1f} px) ;")
    w(f"   écart médian au centre de la bande {em:.1f} px (ligne fixe {eb:.1f} px) ;")
    w(f"   90 % des images sont à moins de {p90m:.1f} px de la bande (ligne fixe {p90b:.1f} px).")

    w("")
    w("3. BOUT EN BOUT : vaisseau détecté ET bien localisé (part des images avec vaisseau)")
    w("-" * 78)
    for key, lab in (("seuil sonde 7", "sonde 7"), ("seuil sonde 9", "sonde 9"), ("seuil global", "ensemble")):
        e0, e5 = d[key][9:11]
        w(f"   {lab:8s}: {pct(e0)} (tolérance 0 px) | {pct(e5)} (tolérance {TOLS[1]} px)")
    w("")
    w(f"PRUDENCE : la sonde 9 n'a que {d['auc sonde 9'][4]} images avec vaisseau ; ses chiffres sont")
    w("très incertains (intervalles de confiance larges).")
    w("=" * 78)
    text = "\n".join(L)
    print("\n" + text)
    path.write_text(text + "\n", encoding="utf-8")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_depth(path, med, dep, base, probe, pos):
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for a, pr in zip(ax, (7, 9)):
        mm = probe[pos] == pr
        a.scatter(med[pos][mm], dep[pos][mm], s=10, alpha=.5, label="modèle")
        a.scatter(med[pos][mm], base[pos][mm], s=6, alpha=.3, c="gray", label="ligne fixe")
        a.plot([0, 130], [0, 130], "r--", lw=1)
        a.set_title(f"sonde {pr} (n={mm.sum()})"); a.set_xlabel("profondeur annotée (px)")
    ax[0].set_ylabel("profondeur prédite (px)"); ax[0].legend(fontsize=8)
    fig.suptitle("Profondeur prédite vs annotée (pointillé : prédiction parfaite)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_curves(path, runs):
    """Loss d'entraînement et de validation (un trait fin par entraînement + moyenne),
    et learning rate de l'encodeur et du décodeur."""
    hist = [h for r in runs for h in r["hist"]]
    lr = runs[0]["lr"]
    if not hist or lr is None:
        return
    plt = _plt()
    from matplotlib.ticker import MaxNLocator
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
    for col, color, lab in (("train_loss_batch_mean", "#2a78d6", "loss entraînement"),
                            ("val_loss", "#eb6834", "loss validation (calibration)")):
        n = min(len(h) for h in hist)
        arr = np.array([h[col].values[:n] for h in hist]); ep = np.arange(1, n + 1)
        for r in arr:
            a1.plot(ep, r, color=color, lw=1, alpha=.15)
        a1.plot(ep, arr.mean(0), color=color, lw=2, marker="o", ms=3, label=f"{lab} (moyenne de {len(arr)})")
    ep = np.arange(1, len(lr) + 1)
    a2.plot(ep, lr[:, 1], drawstyle="steps-mid", marker="o", ms=3, color="#6250d6", lw=2,
            label=f"décodeur + têtes (départ {lr[0, 1]:.0e})")
    a2.plot(ep, lr[:, 0], drawstyle="steps-mid", marker="o", ms=3, color="#1baf7a", lw=2,
            label=f"encodeur (départ {lr[0, 0]:.0e})")
    a2.set_yscale("log")
    for a, t, yl in ((a1, "Loss (présence + profondeur)", "loss"), (a2, "Learning rate", "learning rate (log)")):
        a.grid(axis="y", color="#e6e5e1"); a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.set_xlabel("époque"); a.set_ylabel(yl); a.set_title(t, loc="left", fontsize=10)
        a.xaxis.set_major_locator(MaxNLocator(integer=True)); a.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def evaluate_all(table, cache, splits, seeds, pred_dir, out_dir):
    runs = load_predictions(table, splits, seeds, range(len(splits)), pred_dir)
    rows = [evaluate(f"seed{r['seed']}", r["p"], r["q"], r["cal"], table, splits, cache)[0] for r in runs]
    if len(runs) > 1:
        # Ensemble : moyenne des prédictions ; seuils calibrés sur la moyenne des scores
        # de calibration (on ne moyenne pas les seuils individuels).
        P, Q = np.mean([r["p"] for r in runs], 0), np.mean([r["q"] for r in runs], 0)
        cal = [np.mean([r["cal"][f] for r in runs], 0) for f in range(len(splits))]
        row, det = evaluate("ensemble", P, Q, cal, table, splits, cache, out_dir)
    else:
        r = runs[0]
        row, det = evaluate(f"seed{r['seed']}", r["p"], r["q"], r["cal"], table, splits, cache, out_dir)
        rows = []
    rows.append(row)
    S = pd.DataFrame(rows).set_index("run")
    S.to_csv(out_dir / "resume_final.csv")
    write_bilan(det, S, len(runs), out_dir / "bilan_final.txt")
    plot_curves(out_dir / "courbes_final.png", runs)
    print(f"\nRésultats écrits dans {out_dir}")


# =============================================================================
# 7. PROGRAMME PRINCIPAL
# =============================================================================
def main():
    global EPOCHS
    ap = argparse.ArgumentParser(description="Modèle final : détection et profondeur des vaisseaux")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--evaluer", action="store_true", help="évaluer les prédictions sauvegardées")
    g.add_argument("--entrainer", action="store_true", help="entraîner (reprise automatique) puis évaluer")
    g.add_argument("--test-rapide", action="store_true", help="1 époque, 1 graine, 1 pli : vérification")
    args = ap.parse_args()

    out_dir = FINAL_DIR if not args.test_rapide else FINAL_DIR / "test_rapide"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Tee(out_dir / f"journal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", sys.__stdout__)
    sys.stdout = sys.stderr = log
    try:
        print(f"Modèle final | {ENCODER} ImageNet, {EPOCHS} époques | device {DEV} | torch {torch.__version__}")
        data, table = build_table()
        splits = load_splits(table)
        cache = build_cache(data, table)
        if args.test_rapide:
            EPOCHS = 1
            train_all(table, cache, splits, [0], [0], out_dir / "predictions")
            print("\nTEST RAPIDE OK : données, entraînement et prédiction fonctionnent. "
                  f"Dossier à supprimer : {out_dir}")
            return
        if args.entrainer:
            train_all(table, cache, splits, SEEDS, range(len(splits)), PRED_DIR)
        evaluate_all(table, cache, splits, SEEDS, PRED_DIR, out_dir)
    finally:
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        log.f.close()


if __name__ == "__main__":
    main()
