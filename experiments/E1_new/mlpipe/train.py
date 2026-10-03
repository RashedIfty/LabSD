"""Training and inference for C2 and C3, plus model saving with a manifest."""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .data import FUT, N_CLS, TOKEN_DIM, Sample
from .models import C2AutoBot, C3PlanT, OBJ_DIM, c2_loss


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def device_for(train: bool):
    if train and torch.cuda.is_available():
        return torch.device("cuda")
    if train and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────── C2 ───────────────────────────────

def collate_c2(batch: list[Sample], dev):
    B = len(batch)
    N = max(1, max(len(s.tokens) for s in batch))
    Q = max(1, max(len(s.current) for s in batch))
    tok = torch.zeros(B, N, TOKEN_DIM)
    pad = torch.ones(B, N, dtype=torch.bool)
    cur = torch.zeros(B, Q, dtype=torch.long)
    cpad = torch.ones(B, Q, dtype=torch.bool)
    fut = torch.zeros(B, Q, FUT, 2)
    fm = torch.zeros(B, Q, FUT, dtype=torch.bool)
    for b, s in enumerate(batch):
        n, q = len(s.tokens), len(s.current)
        if n:
            tok[b, :n] = torch.from_numpy(s.tokens); pad[b, :n] = False
        else:
            pad[b, 0] = False          # one zero token so attention is defined
        if q:
            cur[b, :q] = torch.from_numpy(s.current); cpad[b, :q] = False
            fut[b, :q] = torch.from_numpy(s.fut); fm[b, :q] = torch.from_numpy(s.fut_mask)
    fm &= ~cpad.unsqueeze(-1)
    return [t.to(dev) for t in (tok, pad, cur, cpad, fut, fm)]


def _c2_val_ade(model, val, dev, bs=64):
    model.eval(); errs = []
    with torch.no_grad():
        for k in range(0, len(val), bs):
            tok, pad, cur, cpad, fut, fm = collate_c2(val[k:k + bs], dev)
            mu = model(tok, pad, cur, cpad)[0]
            m = fm.unsqueeze(2).float()
            ade = ((torch.linalg.norm(mu - fut.unsqueeze(2), dim=-1) * m).sum(-1) / m.sum(-1).clamp(min=1)).min(-1).values
            ok = fm.any(-1)
            errs.extend(ade[ok].cpu().tolist())
    return float(np.mean(errs)) if errs else float("inf")


def train_c2(samples: list[Sample], val: list[Sample] | None = None, seed=0, epochs=40, bs=32,
             lr=5e-4, patience=8, aug=True, log=print):
    """Early stopping on the validation scenes (Boston), never on the test scenes."""
    from .data import augment
    torch.use_deterministic_algorithms(False)   # MPS training kernels are not deterministic
    seed_all(seed)
    dev = device_for(True)
    model = C2AutoBot().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    data = [s for s in samples if s.fut_mask.any()]
    val = [s for s in (val or []) if s.fut_mask.any()]
    rng = random.Random(seed); nrng = np.random.default_rng(seed)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train(); rng.shuffle(data); stats = []
        for k in range(0, len(data), bs):
            batch = [augment(x, nrng) for x in data[k:k + bs]] if aug else data[k:k + bs]
            tok, pad, cur, cpad, fut, fm = collate_c2(batch, dev)
            mu, sg, rh, lg = model(tok, pad, cur, cpad)
            loss, info = c2_loss(mu, sg, rh, lg, fut, fm, cpad)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            if info: stats.append(info["minADE"])
        sched.step()
        v = _c2_val_ade(model, val, dev) if val else float(np.mean(stats))
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k2: t.detach().cpu().clone() for k2, t in model.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0 or bad == 0:
            log(f"  C2 epoch {ep:3d}  train minADE {np.mean(stats):.3f}  val minADE {v:.3f}{'  *' if bad == 0 else ''}")
        if bad >= patience:
            log(f"  C2 early stop at epoch {ep}, best val minADE {best:.3f}"); break
    model = model.cpu(); model.load_state_dict(best_state)
    model.best_val = best
    return model.eval()


@torch.no_grad()
def predict_c2(model: C2AutoBot, samples: list[Sample], bs=64):
    """Per sample: mu [Q,M,T,2], probs [Q,M] (CPU, deterministic)."""
    torch.use_deterministic_algorithms(True)
    model.eval(); out = []
    for k in range(0, len(samples), bs):
        chunk = samples[k:k + bs]
        tok, pad, cur, cpad, _, _ = collate_c2(chunk, torch.device("cpu"))
        mu, _, _, lg = model(tok, pad, cur, cpad)
        pr = torch.softmax(lg, -1)
        for b, s in enumerate(chunk):
            q = len(s.current)
            out.append((mu[b, :q].numpy(), pr[b, :q].numpy()))
    return out


# ─────────────────────────────── C3 ───────────────────────────────

def c3_objects_from_gt(s: Sample) -> np.ndarray:
    """C3 input in isolated mode: GT objects with their true future (missing steps hold the last position)."""
    rows = []
    for q in range(len(s.current)):
        path = s.fut[q].copy(); last = s.cur_xy[q].copy()
        for t in range(FUT):
            if s.fut_mask[q, t]: last = path[t]
            else: path[t] = last
        oh = np.zeros(N_CLS); oh[s.cur_cls[q]] = 1
        rows.append(np.concatenate([s.cur_xy[q], path.ravel(), oh, [s.cur_conf[q]]]))
    return np.array(rows, np.float32).reshape(-1, OBJ_DIM)


def c3_objects_from_c2(s: Sample, pred) -> np.ndarray:
    """C3 input in pipeline mode: each object's most likely predicted path from C2."""
    mu, pr = pred
    rows = []
    for q in range(len(s.current)):
        path = mu[q, pr[q].argmax()]
        oh = np.zeros(N_CLS); oh[s.cur_cls[q]] = 1
        rows.append(np.concatenate([s.cur_xy[q], path.ravel(), oh, [s.cur_conf[q]]]))
    return np.array(rows, np.float32).reshape(-1, OBJ_DIM)


def ego_vec(s: Sample) -> np.ndarray:
    return np.concatenate([s.ego_hist.ravel(), s.ego_hist_mask.astype(np.float32)]).astype(np.float32)


def _aug_objs(o: np.ndarray, flip: bool, a: float) -> np.ndarray:
    if not len(o):
        return o
    import math
    o = o.copy(); c, sn = math.cos(a), math.sin(a)
    xy = o[:, :2 + FUT * 2].reshape(len(o), -1, 2)
    if flip: xy[..., 1] = -xy[..., 1]
    x, y = xy[..., 0].copy(), xy[..., 1].copy()
    xy[..., 0], xy[..., 1] = c * x - sn * y, sn * x + c * y
    o[:, :2 + FUT * 2] = xy.reshape(len(o), -1)
    return o


def collate_c3(objs: list[np.ndarray], samples: list[Sample], dev, with_target=True):
    B = len(objs); K = max(1, max(len(o) for o in objs))
    x = torch.zeros(B, K, OBJ_DIM); pad = torch.ones(B, K, dtype=torch.bool)
    for b, o in enumerate(objs):
        if len(o):
            x[b, :len(o)] = torch.from_numpy(o); pad[b, :len(o)] = False
    c = torch.tensor([s.command for s in samples], dtype=torch.long)
    e = torch.from_numpy(np.stack([ego_vec(s) for s in samples]))
    y = torch.from_numpy(np.stack([s.ego_fut for s in samples])) if with_target else None
    return x.to(dev), pad.to(dev), c.to(dev), e.to(dev), (y.to(dev) if y is not None else None)


def _c3_val_l2(model, objs, samples, dev, bs=128):
    model.eval(); errs = []
    with torch.no_grad():
        for k in range(0, len(samples), bs):
            x, pad, c, e, y = collate_c3(objs[k:k + bs], samples[k:k + bs], dev)
            errs.extend(torch.linalg.norm(model(x, pad, c, e)[:, -1] - y[:, -1], dim=-1).cpu().tolist())
    return float(np.mean(errs))


def train_c3(objs, samples: list[Sample], val_objs=None, val=None, seed=0, epochs=60, bs=32,
             lr=3e-4, patience=10, aug=True, use_ego=False, log=print):
    from .data import augment
    torch.use_deterministic_algorithms(False)
    seed_all(seed)
    dev = device_for(True)
    model = C3PlanT(use_ego=use_ego).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    idx = list(range(len(samples))); rng = random.Random(seed); nrng = np.random.default_rng(seed + 1)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train(); rng.shuffle(idx); stats = []
        for k in range(0, len(idx), bs):
            b = idx[k:k + bs]
            if aug:
                S = [augment(samples[i], nrng) for i in b]
                O = [_aug_objs(objs[i], *s._aug) for i, s in zip(b, S)]
            else:
                S, O = [samples[i] for i in b], [objs[i] for i in b]
            x, pad, c, e, y = collate_c3(O, S, dev)
            wp = model(x, pad, c, e)
            loss = (wp - y).abs().mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            stats.append(torch.linalg.norm(wp[:, -1] - y[:, -1], dim=-1).mean().item())
        sched.step()
        v = _c3_val_l2(model, val_objs, val, dev) if val else float(np.mean(stats))
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k2: t.detach().cpu().clone() for k2, t in model.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0 or bad == 0:
            log(f"  C3 epoch {ep:3d}  train L2@3s {np.mean(stats):.3f}  val L2@3s {v:.3f}{'  *' if bad == 0 else ''}")
        if bad >= patience:
            log(f"  C3 early stop at epoch {ep}, best val L2@3s {best:.3f}"); break
    model = model.cpu(); model.load_state_dict(best_state)
    model.best_val = best
    return model.eval()


@torch.no_grad()
def plan_c3(model: C3PlanT, objs, samples, bs=128, drop_objects=False):
    torch.use_deterministic_algorithms(True)
    model.eval(); out = []
    for k in range(0, len(samples), bs):
        o = objs[k:k + bs]
        if drop_objects:
            o = [np.zeros((0, OBJ_DIM), np.float32) for _ in o]
        x, pad, c, e, _ = collate_c3(o, samples[k:k + bs], torch.device("cpu"), with_target=False)
        out.append(model(x, pad, c, e).numpy())
    return np.concatenate(out) if out else np.zeros((0, FUT, 2))


# ─────────────────────────────── saving ───────────────────────────────

def save_model(model, path: Path, manifest_path: Path, name: str, info: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    man = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"models": {}}
    man["models"][name] = {"file": path.name, "sha256": h,
                           "params": sum(p.numel() for p in model.parameters()),
                           "saved": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
    manifest_path.write_text(json.dumps(man, indent=1))
    return h


def load_model(cls, path: Path, **kw):
    m = cls(**kw); m.load_state_dict(torch.load(path, map_location="cpu")); return m.eval()
