#!/usr/bin/env python3
"""
sentinel.detect -- find vehicles and people properly.

Motion blobs were enough on synthetic footage, where the only thing that
moved was a car on an empty road. They are not enough on real Ahmedabad
traffic: overlapping rickshaws, headlight glare and a panning PTZ camera
merge into blobs that span the whole frame, and a plate search inside one
of those happily measured the camera's own burned-in caption as a 368px
number plate.

So detection is done with a trained detector, and everything downstream
measures inside a real object box.

Licensing: torchvision, BSD-3-Clause, with COCO weights that ship with it.
Nothing here is AGPL -- Ultralytics YOLO would have been, and in a state
deployment that is a procurement problem rather than a footnote.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

# COCO class ids we care about. 'knife' and 'baseball bat' are the weapon
# classes; the rest are the vehicle and person classes.
COCO = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 6: "bus",
    8: "truck", 44: "bottle", 49: "knife", 39: "baseball bat",
    77: "scissors",
}
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck", "bicycle"}
WEAPON_CLASSES = {"knife", "baseball bat", "scissors"}


@dataclass
class Detection:
    label: str
    score: float
    box: tuple           # x1, y1, x2, y2
    @property
    def width(self) -> int:
        return int(self.box[2] - self.box[0])
    @property
    def height(self) -> int:
        return int(self.box[3] - self.box[1])


class Detector:
    """
    Lazily loaded COCO detector. Kept as a single shared instance because
    loading weights costs seconds and a survey opens thirty cameras.
    """

    _model = None
    _device = None

    @classmethod
    def _load(cls):
        if cls._model is not None:
            return
        import torch
        from torchvision.models.detection import (
            fasterrcnn_mobilenet_v3_large_fpn,
            FasterRCNN_MobileNet_V3_Large_FPN_Weights as W)

        # MobileNet backbone: this runs on thirty cameras on a laptop, and
        # the accuracy we need is "is that a car and how wide is it", not
        # fine-grained classification.
        cls._model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=W.COCO_V1, box_score_thresh=0.35)
        cls._model.eval()
        cls._device = ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
        cls._model.to(cls._device)

    @classmethod
    def device(cls) -> str:
        cls._load()
        return cls._device

    @classmethod
    def detect(cls, frame: np.ndarray, want: Optional[Sequence[str]] = None,
               min_score: float = 0.45) -> List[Detection]:
        import torch
        cls._load()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        with torch.no_grad():
            out = cls._model([t.to(cls._device)])[0]

        res: List[Detection] = []
        for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
            s = float(score)
            if s < min_score:
                continue
            name = COCO.get(int(label))
            if name is None or (want and name not in want):
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            res.append(Detection(name, round(s, 3), (x1, y1, x2, y2)))
        return res

    @classmethod
    def vehicles(cls, frame, min_score: float = 0.45) -> List[Detection]:
        return cls.detect(frame, VEHICLE_CLASSES, min_score)

    @classmethod
    def people(cls, frame, min_score: float = 0.45) -> List[Detection]:
        return cls.detect(frame, ("person",), min_score)


def osd_mask(shape, top_frac: float = 0.09, bottom_frac: float = 0.12):
    """
    Region of the frame occupied by burned-in text.

    Every camera on this grid stamps a clock across the top and a caption
    across the bottom. Both are bright rectangles full of dark characters,
    which is exactly what a plate looks like to a detector -- the caption
    "bhai Bridge" was measured as a number plate. Analytics ignore these
    bands.
    """
    h, w = shape[:2]
    m = np.ones((h, w), np.uint8) * 255
    m[:int(h * top_frac), :] = 0
    m[int(h * (1 - bottom_frac)):, :] = 0
    return m
