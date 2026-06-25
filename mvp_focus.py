from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights, MobileNet_V2_Weights


TURKEY_TZ = ZoneInfo("Europe/Istanbul")


@dataclass
class FocusPrediction:
    timestamp: str
    label: str
    focused_probability: float
    frame_focus_score: float
    window_focus_score: float
    status: str
    alert: str
    face_found: bool


def build_model(num_classes: int = 2, backbone: str = "efficientnet_b0") -> nn.Module:
    if backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if backbone == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported backbone for MVP: {backbone}")


class RollingFocusMonitor:
    def __init__(
        self,
        window_size: int = 10,
        min_window_size: int = 5,
        warning_threshold: float = 50.0,
        soft_warning_threshold: float = 60.0,
    ) -> None:
        self.window_size = window_size
        self.min_window_size = min_window_size
        self.warning_threshold = warning_threshold
        self.soft_warning_threshold = soft_warning_threshold
        self.scores: Deque[float] = deque(maxlen=window_size)

    def update(self, frame_focus_score: float, timestamp: datetime) -> tuple[float, str, str]:
        self.scores.append(frame_focus_score)
        window_focus_score = float(np.mean(self.scores))

        if len(self.scores) < self.min_window_size:
            return window_focus_score, "Veri Toplaniyor", ""

        clock = timestamp.strftime("%H:%M:%S")
        if window_focus_score < self.warning_threshold:
            return (
                window_focus_score,
                "Dusuk Odak",
                f"Saat {clock} civarinda odak seviyeniz dusuk seviyedeydi.",
            )
        if window_focus_score < self.soft_warning_threshold:
            return (
                window_focus_score,
                "Ortalama Odak",
                f"Saat {clock} civarinda odak seviyeniz ortalama seviyedeydi.",
            )
        return (
            window_focus_score,
            "Yuksek Odak",
            f"Saat {clock} civarinda odak seviyeniz yuksek seviyedeydi.",
        )


class FaceCropper:
    def __init__(self, padding: float = 0.25, min_face_size: int = 60) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(str(cascade_path))
        if self.detector.empty():
            raise RuntimeError(f"Face detector could not be loaded: {cascade_path}")
        self.padding = padding
        self.min_face_size = min_face_size

    def crop(self, image: Image.Image) -> Image.Image | None:
        rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )
        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda box: int(box[2]) * int(box[3]))
        pad_x = int(w * self.padding)
        pad_y = int(h * self.padding)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(rgb.shape[1], x + w + pad_x)
        y2 = min(rgb.shape[0], y + h + pad_y)
        return image.crop((x1, y1, x2, y2))


class FocusModel:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        self.backbone = checkpoint.get("backbone", "efficientnet_b0")
        self.num_classes = int(checkpoint.get("num_classes", 2))
        raw_mapping = checkpoint.get("class_mapping", {0: "low_focus", 1: "focused"})
        self.class_mapping = {int(key): value for key, value in raw_mapping.items()}
        self.threshold = float(checkpoint.get("video_threshold", checkpoint.get("threshold", 0.5)))

        self.model = build_model(self.num_classes, self.backbone).to(self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("model_state", checkpoint))
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> tuple[str, float, float]:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probabilities = F.softmax(self.model(tensor), dim=1).squeeze(0).detach().cpu().numpy()
        focused_probability = float(probabilities[1])
        frame_focus_score = focused_probability * 100.0
        predicted_index = int(np.argmax(probabilities))
        label = str(self.class_mapping.get(predicted_index, predicted_index))
        return label, focused_probability, frame_focus_score
