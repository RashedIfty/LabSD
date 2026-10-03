"""ML prediction module C2 (AutoBot-Ego style) and ML planning module C3 (PlanT style).

C2 follows AutoBot-Ego (Girgis et al., ICLR 2022): a transformer encoder over the
scene context, learned per-mode seeds that attend to the context, a bivariate
Gaussian output per future step, and a mode-probability head. Two changes for
this study: the context is the unordered set of detections of the last four
keyframes (no track identities, the association is learned by attention, as in
Weng et al., CVPR 2022), and there is no map input.

C3 follows PlanT (Renz et al., CoRL 2022): one token per object, a [CLS] token,
a transformer encoder, and a GRU that rolls out the waypoints. Changes for
nuScenes without maps: the route tokens are replaced by a driving command token
(left / straight / right, as in VAD and AD-MLP), and C3 receives no ego speed
and no ego history (Zhai et al., Li et al.: with ego status a planning module
can ignore the objects).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F_

from .data import FUT, N_CLS, TOKEN_DIM

POS_SCALE = 50.0     # metres -> network units


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))


REL_DIM = 2 + 1 + 1 + N_CLS + 1 + 1     # offset, distance, time offset, class, same class, confidence
REL_SCALE = 5.0                          # metres per unit for offsets seen from the target


class C2AutoBot(nn.Module):
    """AutoBot-Ego style: every object of keyframe i is the target in turn.

    As in AutoBot-Ego, the scene is seen from the target: each detection of the
    last four keyframes is described relative to the target (offset, distance,
    time offset, class, same class as the target, confidence). A transformer
    encoder runs over this target-centred set, so finding the target's own
    earlier detections (and with them its motion) is learned by attention. The
    decoder has learned mode seeds that attend to the set and output a
    bivariate Gaussian per future step plus mode probabilities.
    """

    def __init__(self, d=128, heads=4, enc_layers=2, dec_layers=2, modes=6, dropout=0.1):
        super().__init__()
        self.modes = modes
        self.rel = _mlp(REL_DIM, d, d)
        self.target = _mlp(2 + N_CLS + 1, d, d)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True), enc_layers,
            enable_nested_tensor=False)
        self.mode_seeds = nn.Parameter(torch.randn(modes, d) * 0.02)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d, heads, 2 * d, dropout, batch_first=True), dec_layers)
        self.out = _mlp(d, d, FUT * 5)
        self.prob = _mlp(d, d, 1)

    def forward(self, tokens, tok_pad, cur_idx, cur_pad):
        """tokens [B,N,F], tok_pad [B,N] True=padding, cur_idx [B,Q], cur_pad [B,Q].

        Returns mu [B,Q,M,T,2] (metres, ego frame), sigma [B,Q,M,T,2], rho [B,Q,M,T],
        logits [B,Q,M].
        """
        B, N, F = tokens.shape
        Q = cur_idx.size(1)
        cur = torch.gather(tokens, 1, cur_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, F))   # [B,Q,F]
        rel = tokens[:, None, :, :2] - cur[:, :, None, :2]                                   # [B,Q,N,2]
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
        cls_tok = tokens[..., 2:2 + N_CLS]
        same = (cls_tok[:, None] * cur[:, :, None, 2:2 + N_CLS]).sum(-1, keepdim=True)
        feat = torch.cat([(rel / REL_SCALE).clamp(-10, 10), (dist / REL_SCALE).clamp(max=10),
                          tokens[:, None, :, -1:].expand(-1, Q, -1, -1),
                          cls_tok[:, None].expand(-1, Q, -1, -1), same,
                          tokens[:, None, :, -2:-1].expand(-1, Q, -1, -1)], -1)            # [B,Q,N,REL_DIM]
        mem_pad = tok_pad[:, None, :].expand(-1, Q, -1).reshape(B * Q, N)
        mem = self.encoder(self.rel(feat).view(B * Q, N, -1), src_key_padding_mask=mem_pad)
        tgt = self.target(torch.cat([cur[..., :2] / POS_SCALE, cur[..., 2:2 + N_CLS], cur[..., -2:-1]], -1))
        q = (tgt.unsqueeze(2) + self.mode_seeds.view(1, 1, self.modes, -1)).view(B * Q, self.modes, -1)
        dec = self.decoder(q, mem, memory_key_padding_mask=mem_pad).view(B, Q, self.modes, -1)
        p = self.out(dec).view(B, Q, self.modes, FUT, 5)
        mu = cur[..., :2].view(B, Q, 1, 1, 2) + torch.cumsum(p[..., :2], dim=3) * 2.0
        sigma = F_.softplus(p[..., 2:4]) + 0.05
        rho = torch.tanh(p[..., 4]) * 0.9
        logits = self.prob(dec).squeeze(-1)
        return mu, sigma, rho, logits


def c2_loss(mu, sigma, rho, logits, fut, mask, cur_pad, ade_weight=1.0):
    """Winner-takes-all: Gaussian NLL and ADE/FDE on the closest mode + mode classification.

    Objects with no valid future step (false detections, or objects that leave)
    contribute nothing.
    """
    m = mask.unsqueeze(2).float()                                   # [B,Q,1,T]
    err = torch.linalg.norm(mu - fut.unsqueeze(2), dim=-1)          # [B,Q,M,T]
    nvalid = m.sum(-1).clamp(min=1)
    ade = (err * m).sum(-1) / nvalid                                # [B,Q,M]
    best = ade.argmin(-1)                                           # [B,Q]
    valid_obj = (mask.any(-1) & ~cur_pad)                           # [B,Q]
    if valid_obj.sum() == 0:
        return mu.sum() * 0.0, {}
    sel = lambda t: t.gather(2, best.view(*best.shape, 1, *([1] * (t.dim() - 3))).expand(
        *best.shape, 1, *t.shape[3:])).squeeze(2)
    mu_b, sg_b, rh_b = sel(mu), sel(sigma), sel(rho)                # [B,Q,T,2], [B,Q,T]
    d = fut - mu_b
    zx, zy = d[..., 0] / sg_b[..., 0], d[..., 1] / sg_b[..., 1]
    one_r = (1 - rh_b ** 2).clamp(min=1e-4)
    nll = (torch.log(sg_b[..., 0] * sg_b[..., 1]) + 0.5 * torch.log(one_r)
           + (zx ** 2 + zy ** 2 - 2 * rh_b * zx * zy) / (2 * one_r))
    nll = (nll * mask).sum(-1) / nvalid.squeeze(-1)
    fde_idx = (mask.float() * torch.arange(1, mask.size(-1) + 1, device=mask.device)).argmax(-1)
    fde = err.gather(2, best.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, err.size(-1))).squeeze(2)
    fde = fde.gather(-1, fde_idx.unsqueeze(-1)).squeeze(-1)
    ce = F.cross_entropy(logits[valid_obj], best[valid_obj])
    ade_b = ade.gather(-1, best.unsqueeze(-1)).squeeze(-1)
    loss = (nll[valid_obj].mean() + ce
            + ade_weight * (ade_b[valid_obj].mean() + fde[valid_obj].mean()) / POS_SCALE * 10)
    return loss, {"nll": nll[valid_obj].mean().item(), "minADE": ade_b[valid_obj].mean().item(),
                  "ce": ce.item()}


OBJ_DIM = 2 + FUT * 2 + N_CLS + 1      # current x,y; 6-step path; class; confidence


EGO_DIM = 3 * 2 + 3                   # ego positions at the 3 previous keyframes + their mask


class C3PlanT(nn.Module):
    """use_ego=False: the report's design (no ego speed or history).
    use_ego=True: one extra token with the ego vehicle's past path (last 1.5 s)."""

    def __init__(self, d=128, heads=4, layers=4, dropout=0.1, use_ego=False):
        super().__init__()
        self.use_ego = use_ego
        self.obj = _mlp(OBJ_DIM, d, d)
        self.ego = _mlp(EGO_DIM, d, d) if use_ego else None
        self.type_emb = nn.Embedding(4, d)        # 0 = CLS, 1 = object, 2 = command, 3 = ego
        self.cmd = nn.Embedding(3, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True), layers,
            enable_nested_tensor=False)
        self.gru = nn.GRUCell(2 + 3, d)
        self.head = _mlp(d, d, 2)

    def forward(self, objs, obj_pad, command, ego=None):
        """objs [B,K,OBJ_DIM] (metres), obj_pad [B,K], command [B], ego [B,EGO_DIM] -> waypoints [B,FUT,2]."""
        B = objs.size(0)
        x = objs.clone()
        x[..., : 2 + FUT * 2] = x[..., : 2 + FUT * 2] / POS_SCALE
        o = self.obj(x) + self.type_emb.weight[1]
        c = (self.cmd(command) + self.type_emb.weight[2]).unsqueeze(1)
        cls = self.cls.expand(B, -1, -1) + self.type_emb.weight[0]
        head = [cls, c]
        if self.use_ego:
            e = ego.clone(); e[:, :6] = e[:, :6] / POS_SCALE
            head.append((self.ego(e) + self.type_emb.weight[3]).unsqueeze(1))
        seq = torch.cat(head + [o], 1)
        pad = torch.cat([torch.zeros(B, len(head), dtype=torch.bool, device=objs.device), obj_pad], 1)
        h = self.encoder(seq, src_key_padding_mask=pad)[:, 0]
        cmd_oh = F.one_hot(command, 3).float()
        wp = torch.zeros(B, 2, device=objs.device)
        out = []
        for _ in range(FUT):
            h = self.gru(torch.cat([wp / POS_SCALE, cmd_oh], -1), h)
            wp = wp + self.head(h) * 5.0
            out.append(wp)
        return torch.stack(out, 1)
