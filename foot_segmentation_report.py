import json
import os
from typing import Optional

import cv2
import numpy as np

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except Exception:
    SAM_AVAILABLE = False

from utils import detect_credit_card_reference


# -------------------- SAM helpers --------------------

def initialize_sam(model_type: str = "vit_b", checkpoint: str = "sam_vit_b_01ec64.pth") -> Optional[SamAutomaticMaskGenerator]:
    """Initialize SAM if possible. Returns a mask generator or None."""
    if not SAM_AVAILABLE:
        return None
    if not os.path.exists(checkpoint):
        print("SAM checkpoint not found, segmentation will be mocked.")
        return None
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    return SamAutomaticMaskGenerator(model=sam)


# -------------------- Segmentation --------------------

def segment_image(image: np.ndarray, generator: Optional[SamAutomaticMaskGenerator]) -> np.ndarray:
    """Return a binary mask for the foot using SAM or a fallback."""
    if generator is not None:
        masks = generator.generate(image)
        if masks:
            best = max(masks, key=lambda m: m["area"])
            return best["segmentation"].astype(np.uint8) * 255
    # Fallback segmentation (simple threshold)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


# -------------------- Measurements --------------------

def _largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def measure_top_view(image: np.ndarray, mask: np.ndarray, px_to_cm: float) -> dict:
    """Compute foot length, forefoot width and hallux valgus angle."""
    contour = _largest_contour(mask)
    if contour is None:
        return {}
    x, y, w, h = cv2.boundingRect(contour)
    length = h * px_to_cm
    width = w * px_to_cm

    pts = contour[:, 0, :]
    heel = pts[pts[:, 1].argmax()]
    toe_band = pts[pts[:, 1] == pts[:, 1].min()]
    if len(toe_band) >= 2:
        center_toe = toe_band.mean(axis=0)
        big_toe = toe_band[toe_band[:, 0].argmin()]
    else:
        center_toe = pts[pts[:, 1].argmin()]
        big_toe = center_toe

    v1 = center_toe - heel
    v2 = big_toe - heel
    ang = 0.0
    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_a = float(np.clip(cos_a, -1.0, 1.0))
        ang = np.degrees(np.arccos(cos_a))

    return {
        "foot_length_cm": round(length, 2),
        "forefoot_width_cm": round(width, 2),
        "hallux_valgus_angle_deg": round(ang, 2),
    }


def measure_bottom_view(image: np.ndarray, mask: np.ndarray, px_to_cm: float) -> dict:
    contour = _largest_contour(mask)
    if contour is None:
        return {}
    area = cv2.contourArea(contour) * (px_to_cm ** 2)
    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = (w * px_to_cm) * (h * px_to_cm)

    pts = contour[:, 0, :]
    foot_length = h
    y_start = int(y + foot_length * 0.3)
    y_end = int(y + foot_length * 0.7)
    max_w = 0
    min_w = 1e9
    for yy in range(y_start, y_end):
        row = np.where(mask[yy] > 0)[0]
        if len(row) > 1:
            w_row = row[-1] - row[0]
            max_w = max(max_w, w_row)
            min_w = min(min_w, w_row)
    arch_ratio = min_w / max_w if max_w else 0
    if arch_ratio > 0.75:
        arch_type = "flat"
    elif arch_ratio > 0.5:
        arch_type = "normal"
    else:
        arch_type = "high"
    return {
        "footprint_area_cm2": round(area, 2),
        "contact_ratio": round(area / bbox_area if bbox_area else 0, 2),
        "arch_type": arch_type,
    }


def measure_side_view(image: np.ndarray, mask: np.ndarray, px_to_cm: float) -> dict:
    contour = _largest_contour(mask)
    if contour is None:
        return {}
    x, y, w, h = cv2.boundingRect(contour)
    bottom = y + h
    pts = contour[:, 0, :]
    x_start = int(x + w * 0.4)
    x_end = int(x + w * 0.6)
    region = pts[(pts[:, 0] >= x_start) & (pts[:, 0] <= x_end)]
    if len(region) == 0:
        instep_h = 0
    else:
        instep_h = bottom - region[:, 1].min()
    arch_h = bottom - pts[:, 1].min()
    return {
        "instep_height_cm": round(instep_h * px_to_cm, 2),
        "arch_height_cm": round(arch_h * px_to_cm, 2),
    }


def measure_back_view(image: np.ndarray, mask: np.ndarray, px_to_cm: float) -> dict:
    contour = _largest_contour(mask)
    if contour is None:
        return {}
    x, y, w, h = cv2.boundingRect(contour)
    pts = contour[:, 0, :]
    y_start = int(y + h * 0.8)
    region = pts[pts[:, 1] >= y_start]
    if len(region) == 0:
        width = 0
    else:
        width = region[:, 0].max() - region[:, 0].min()
    return {"heel_width_cm": round(width * px_to_cm, 2)}


# -------------------- Utility --------------------

def pixel_to_cm_scale(image: np.ndarray) -> float:
    result = detect_credit_card_reference(image)
    if result.get("success"):
        px_per_cm = result["ratio_px_mm"] * 10
        return 1.0 / px_per_cm
    print("Reference object not found; assuming 1px=1cm")
    return 1.0


def main(top, bottom, side, back, checkpoint="sam_vit_b_01ec64.pth"):
    generator = initialize_sam(checkpoint=checkpoint)
    images = {"top": cv2.imread(top), "bottom": cv2.imread(bottom), "side": cv2.imread(side), "back": cv2.imread(back)}
    if any(v is None for v in images.values()):
        raise FileNotFoundError("One or more images could not be loaded")

    scale = pixel_to_cm_scale(images["top"])

    results = {}
    masks = {}
    for key, img in images.items():
        masks[key] = segment_image(img, generator)

    results.update(measure_top_view(images["top"], masks["top"], scale))
    results.update(measure_bottom_view(images["bottom"], masks["bottom"], scale))
    results.update(measure_side_view(images["side"], masks["side"], scale))
    results.update(measure_back_view(images["back"], masks["back"], scale))

    with open("foot_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved report to foot_report.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Foot report using SAM")
    parser.add_argument("top")
    parser.add_argument("bottom")
    parser.add_argument("side")
    parser.add_argument("back")
    parser.add_argument("--checkpoint", default="sam_vit_b_01ec64.pth")
    args = parser.parse_args()

    main(args.top, args.bottom, args.side, args.back, args.checkpoint)
