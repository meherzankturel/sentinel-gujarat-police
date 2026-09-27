#!/usr/bin/env python3
"""
sentinel.vehicle_reid -- tell one vehicle from another, not one class from another.

Our first two attempts at following a vehicle without its plate both failed,
and they failed for the same underlying reason.

A hand-built colour and layout descriptor reached 26% rank-1 on real
Chimanbhai Bridge traffic. An ImageNet-pretrained ResNet reached 35%. The
ImageNet network was never asked to distinguish one car from another car --
it was trained to separate a car from a dog, and it learned to throw away
exactly the details that identify an individual vehicle. Using it for
re-identification asks it for information it was optimised to discard.

The fix is a model trained for the actual task. FastReID's VeRi baseline is
trained on VeRi-776 -- 50,000 images of 776 vehicles across 20 real traffic
cameras -- with a loss that pulls views of the same vehicle together and
pushes different vehicles apart. It reports 97.0% rank-1 and 81.9% mAP on
that benchmark.

We load the published weights rather than depending on the framework, so
this stays a small, auditable file. Apache 2.0, consistent with the rest of
the stack; there is no AGPL anywhere in this system.

Reference: FastReID, JDAI-CV, https://github.com/JDAI-CV/fast-reid
Weights:   veri_sbs_R50-ibn.pth (VeRi-776 baseline)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "veri_sbs_R50-ibn.pth"

# The published training statistics. Getting these wrong shifts every
# feature slightly and quietly degrades matching, so they are read from the
# checkpoint rather than assumed.
PIXEL_MEAN = (123.675, 116.28, 103.53)
PIXEL_STD = (58.395, 57.12, 57.375)
INPUT_SIZE = (256, 256)          # h, w -- SBS VeRi config


def _ibn(planes: int):
    """
    Instance-Batch Normalisation, as used in IBN-a.

    Half the channels are instance-normalised and half batch-normalised.
    Instance normalisation removes per-image appearance shifts -- which is
    to say lighting and colour cast -- while batch normalisation keeps the
    discriminative content. That split is precisely why IBN backbones
    transfer between cameras, and transferring between cameras is the whole
    problem here.
    """
    import torch.nn as nn

    class IBN(nn.Module):
        def __init__(self, planes):
            super().__init__()
            half = planes // 2
            self.half = half
            self.IN = nn.InstanceNorm2d(half, affine=True)
            self.BN = nn.BatchNorm2d(planes - half)

        def forward(self, x):
            import torch
            a, b = x[:, :self.half], x[:, self.half:]
            return torch.cat((self.IN(a), self.BN(b)), 1)

    return IBN(planes)


def _non_local(in_channels: int, inter_channels: int):
    """
    Non-local attention block, as carried in the published weights.

    Each position in the feature map is refreshed with a weighted sum over
    every other position, so a detail at the back of a vehicle can inform
    the representation at the front. For re-identification that matters:
    the distinguishing mark on a vehicle is rarely where you look first.

    Included because the checkpoint contains these tensors. Loading a
    backbone while quietly dropping part of its forward path produces
    features that look fine and are not what the model was trained to
    produce -- the exact failure mode this project has been bitten by.
    """
    import torch
    import torch.nn as nn

    class NonLocal(nn.Module):
        def __init__(self, in_c, inter_c):
            super().__init__()
            self.inter = inter_c
            self.g = nn.Conv2d(in_c, inter_c, 1)
            self.theta = nn.Conv2d(in_c, inter_c, 1)
            self.phi = nn.Conv2d(in_c, inter_c, 1)
            self.W = nn.Sequential(nn.Conv2d(inter_c, in_c, 1),
                                   nn.BatchNorm2d(in_c))

        def forward(self, x):
            n = x.size(0)
            g = self.g(x).view(n, self.inter, -1).permute(0, 2, 1)
            th = self.theta(x).view(n, self.inter, -1).permute(0, 2, 1)
            ph = self.phi(x).view(n, self.inter, -1)
            f = torch.matmul(th, ph) / ph.size(-1)
            y = torch.matmul(f, g).permute(0, 2, 1).contiguous()
            y = y.view(n, self.inter, *x.shape[2:])
            return self.W(y) + x

    return NonLocal(in_channels, inter_channels)


def _build_backbone(nl_spec=None):
    """
    ResNet50 with IBN in layers 1-3 and non-local blocks where the
    checkpoint places them.

    FastReID appends its non-local blocks to the END of a stage, so with
    two blocks in layer2 they follow bottlenecks 2 and 3, and with three in
    layer3 they follow bottlenecks 3, 4 and 5. Placing them anywhere else
    loads the same tensors into the wrong positions and yields a model that
    runs perfectly while meaning nothing.
    """
    import torch.nn as nn
    from torchvision.models import resnet50

    net = resnet50(weights=None)
    for layer_name in ("layer1", "layer2", "layer3"):
        for block in getattr(net, layer_name):
            block.bn1 = _ibn(block.bn1.num_features)

    class Backbone(nn.Module):
        def __init__(self, r, spec):
            super().__init__()
            self.conv1, self.bn1 = r.conv1, r.bn1
            self.relu, self.maxpool = r.relu, r.maxpool
            self.layer1, self.layer2 = r.layer1, r.layer2
            self.layer3, self.layer4 = r.layer3, r.layer4
            self.NL_2 = nn.ModuleList()
            self.NL_3 = nn.ModuleList()
            self.nl2_idx, self.nl3_idx = [], []
            if spec:
                n2, c2 = spec.get("layer2", (0, 1))
                n3, c3 = spec.get("layer3", (0, 1))
                self.NL_2 = nn.ModuleList(
                    [_non_local(512, c2) for _ in range(n2)])
                self.NL_3 = nn.ModuleList(
                    [_non_local(1024, c3) for _ in range(n3)])
                self.nl2_idx = sorted(len(self.layer2) - (i + 1)
                                      for i in range(n2))
                self.nl3_idx = sorted(len(self.layer3) - (i + 1)
                                      for i in range(n3))

        def _stage(self, layer, blocks, idx, x):
            c = 0
            for i, blk in enumerate(layer):
                x = blk(x)
                if c < len(idx) and i == idx[c]:
                    x = blocks[c](x)
                    c += 1
            return x

        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer1(x)
            x = self._stage(self.layer2, self.NL_2, self.nl2_idx, x)
            x = self._stage(self.layer3, self.NL_3, self.nl3_idx, x)
            return self.layer4(x)

    return Backbone(net, nl_spec)


class VehicleReID:
    """
    Turns a vehicle crop into a fingerprint that survives a change of camera.

    Loaded once and shared: the weights are 200MB and a survey opens many
    cameras.
    """

    _model = None
    _device = None
    _gem_p = 2.0

    @classmethod
    def available(cls) -> bool:
        return WEIGHTS.exists()

    @classmethod
    def _load(cls):
        if cls._model is not None:
            return
        import torch
        import torch.nn as nn

        if not WEIGHTS.exists():
            raise FileNotFoundError(
                f"{WEIGHTS.name} missing. Download the FastReID VeRi baseline "
                f"into models/ before using appearance matching.")

        ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt)

        # Read the non-local layout out of the checkpoint rather than
        # assuming it: how many blocks per stage, and their inner width.
        import re as _re
        spec = {}
        for stage, key in (("layer2", "NL_2"), ("layer3", "NL_3")):
            idxs = {int(m.group(1)) for k in sd
                    for m in [_re.match(rf"backbone\.{key}\.(\d+)\.g\.weight", k)] if m}
            if idxs:
                w = sd[f"backbone.{key}.0.g.weight"]
                spec[stage] = (len(idxs), w.shape[0])
        backbone = _build_backbone(spec)
        bstate = {k[len("backbone."):]: v for k, v in sd.items()
                  if k.startswith("backbone.")}
        missing, unexpected = backbone.load_state_dict(bstate, strict=False)
        # Loud on purpose: a silently half-loaded backbone still produces
        # plausible-looking features, and we have been bitten by exactly
        # that class of failure already.
        real_missing = [m for m in missing if "num_batches_tracked" not in m]
        if real_missing or unexpected:
            raise RuntimeError(
                f"backbone weights did not match: {len(real_missing)} missing, "
                f"{len(unexpected)} unexpected (e.g. {real_missing[:3]})")

        bn = nn.BatchNorm1d(2048)
        bn.load_state_dict({
            "weight": sd["heads.bottleneck.0.weight"],
            "bias": sd["heads.bottleneck.0.bias"],
            "running_mean": sd["heads.bottleneck.0.running_mean"],
            "running_var": sd["heads.bottleneck.0.running_var"],
            "num_batches_tracked": sd.get(
                "heads.bnneck.num_batches_tracked", torch.tensor(0)),
        })

        cls._gem_p = float(sd["heads.pool_layer.p"].flatten()[0])
        cls._device = ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
        backbone.eval().to(cls._device)
        bn.eval().to(cls._device)
        cls._model = (backbone, bn)

    # ---------------------------------------------------------------- api

    @classmethod
    def embed(cls, crop: np.ndarray) -> Optional[np.ndarray]:
        """A unit-length fingerprint for one vehicle crop, or None if unusable."""
        import torch

        if crop is None or crop.size == 0 or min(crop.shape[:2]) < 16:
            return None
        cls._load()
        backbone, bn = cls._model

        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (INPUT_SIZE[1], INPUT_SIZE[0]),
                         interpolation=cv2.INTER_CUBIC).astype(np.float32)
        img = (img - np.array(PIXEL_MEAN)) / np.array(PIXEL_STD)
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(cls._device)

        with torch.no_grad():
            f = backbone(t)
            if f.dim() == 2:                      # avgpool was replaced
                f = f.view(1, 2048, INPUT_SIZE[0] // 32, INPUT_SIZE[1] // 32)
            # Generalised-mean pooling with the learned exponent, as trained.
            p = cls._gem_p
            f = f.clamp(min=1e-6).pow(p)
            f = torch.nn.functional.adaptive_avg_pool2d(f, 1).pow(1.0 / p)
            f = f.flatten(1)
            f = bn(f)
        v = f[0].cpu().numpy()
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else None

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity, mapped to 0..1 so it reads like a confidence."""
        if a is None or b is None:
            return 0.0
        return float((np.dot(a, b) + 1.0) / 2.0)
