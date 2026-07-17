#!/usr/bin/env python3
"""VAE-Vita V1 Evaluation -- Comprehensive SIC-POVM Analysis"""
import torch
import numpy as np
import json, sys, time

sys.path.insert(0, '/home/mrnob0dy666/imsgct/ig-pulse/vae_vita')
from hyperspherical_vae import HypersphericalVAE
from crystal_data import CrystalDataset

print("=" * 60)
print("VAE-Vita V1 Comprehensive Evaluation")
print("=" * 60)

# Load model
ckpt = torch.load('vae_vita_sic50.pt', map_location='cpu')
model = HypersphericalVAE(d_latent=12, hidden_dim=512, beta=0.2, lambda_sic=50.0)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print(f"Model loaded on {device}")

ds = CrystalDataset()
rng = np.random.default_rng(42)
results = {}

# 1. Accuracy
N, bs = 20000, 1024
correct, total = 0, 0
all_latents = []

t0 = time.time()
with torch.no_grad():
    for b_start in range(0, N, bs):
        bsize = min(bs, N - b_start)
        indices = rng.integers(0, ds.total, size=bsize)
        x_batch = np.stack([ds.get_ordinal(i) for i in indices])
        x_t = torch.from_numpy(x_batch).float().to(device)
        mu, kappa, z, logits = model(x_t)
        all_latents.append(z.cpu())
        for i, (logit, nv) in enumerate(zip(logits, [4,5,4,5,3,5,3,4,5,4,3,4])):
            preds = logit.argmax(dim=-1)
            targets = (x_t[:, i] * (nv - 1)).long()
            correct += (preds == targets).sum().item()
            total += targets.shape[0]

accuracy = correct / total
results['accuracy'] = float(accuracy)
print(f"\nAccuracy: {accuracy*100:.2f}% ({total} predictions)")
print(f"Random baseline: ~24.8%")
t1 = time.time()
print(f"Time: {t1-t0:.1f}s")

# 2. Per-primitive accuracy
print(f"\nPer-primitive accuracy:")
prim_names = ['D','T','R','P','F','K','G','Gamma','Phi','H','S','Omega']
prim_sizes = [4,5,4,5,3,5,3,4,5,4,3,4]
with torch.no_grad():
    indices = rng.integers(0, ds.total, size=5000)
    x_batch = np.stack([ds.get_ordinal(i) for i in indices])
    x_t = torch.from_numpy(x_batch).float().to(device)
    mu, kappa, z, logits = model(x_t)
    for i, (logit, nv, pname) in enumerate(zip(logits, prim_sizes, prim_names)):
        preds = logit.argmax(dim=-1)
        targets = (x_t[:, i] * (nv - 1)).long()
        acc = (preds == targets).float().mean().item()
        print(f'  {pname:>6s} ({nv} vals): {acc*100:.1f}%')

# 3. SIC metrics
z_all = torch.cat(all_latents, dim=0)
N_z = len(z_all)
overlaps = torch.mm(z_all, z_all.T)
overlaps_sq = overlaps ** 2
mask = ~torch.eye(N_z, dtype=torch.bool, device=z_all.device)
off_diag = overlaps_sq[mask]
mean_off = off_diag.mean().item()
std_off = off_diag.std().item()

print(f"\nSIC Analysis ({N_z} samples):")
print(f'  Mean overlap (off-diag): {mean_off:.6f}')
print(f'  Std  overlap (off-diag): {std_off:.6f}')
print(f'  SIC target:              1/13 = {1/13:.6f}')
print(f'  Uniform expectation:      1/12 = {1/12:.6f}')
print(f'  Deviation from uniform:  {abs(mean_off - 1/12):.6f}')
print(f'  Deviation from SIC:      {abs(mean_off - 1/13):.6f}')

results['mean_overlap'] = mean_off
results['std_overlap'] = std_off
results['target_sic'] = 1/13
results['target_uniform'] = 1/12
results['deviation_sic'] = abs(mean_off - 1/13)
results['deviation_uniform'] = abs(mean_off - 1/12)

# 4. Kappa statistics
with torch.no_grad():
    indices = rng.integers(0, ds.total, size=5000)
    x_batch = np.stack([ds.get_ordinal(i) for i in indices])
    x_t = torch.from_numpy(x_batch).float().to(device)
    mu, kappa, z, logits = model(x_t)
    print(f"\nKappa statistics:")
    print(f'  Mean kappa: {kappa.mean().item():.2f}')
    print(f'  Min kappa:  {kappa.min().item():.2f}')
    print(f'  Max kappa:  {kappa.max().item():.2f}')
    print(f'  % saturated (≥49.9): {(kappa >= 49.9).float().mean().item()*100:.1f}%')
    results['mean_kappa'] = float(kappa.mean().item())
    results['pct_saturated'] = float((kappa >= 49.9).float().mean().item())

# Save
with open('/home/mrnob0dy666/imsgct/vae_vita_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to vae_vita_analysis.json")
print("=" * 60)
