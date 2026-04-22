import base64
import io
import logging
import os
import time
from typing import Any, Dict, Literal, Optional, Tuple

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageOps

load_dotenv()

logger = logging.getLogger(__name__)

BRIA_API_KEY = os.getenv("BRIA_API_KEY")
BRIA_BASE_URL = "https://engine.prod.bria-api.com"

REMOVE_BG_URL = f"{BRIA_BASE_URL}/v2/image/edit/remove_background"
ENHANCE_URL = f"{BRIA_BASE_URL}/v2/image/edit/enhance"
INCREASE_RESOLUTION_URL = f"{BRIA_BASE_URL}/v2/image/edit/increase_resolution"

# premium_3d uses GPT-image-1.5 after BRIA cleanup
BriaMode = Literal["none", "enhance", "increase_resolution", "premium_3d"]

POLL_INTERVAL_SECONDS = 2
POLL_TIMEOUT_SECONDS = 180

MAX_INPUT_MP = 50  # megapixels

# ----------------------------
# GPT-image-1.5 config
# ----------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_GPT_PROMPT = """
You are a professional fashion e-commerce photographer.

INPUT: A cleanly extracted product image on a transparent background.

YOUR JOB:
- Present the garment as a natural, self-supporting 3D form
- Remove any remaining hanger or mannequin artifacts
- Pure white background (#FFFFFF) only
- Soft even studio lighting — no harsh shadows, no gradients, no glow
- The garment should look naturally draped as if on an invisible form
- Preserve the EXACT fabric pattern, print, color, and all design details
- Preserve all features: bows, ties, straps, buttons, embroidery, stitching exactly as they appear
- Square 1:1 composition
- Full garment visible from top to bottom — do not crop any part
- Product centered with equal padding on all sides
- No body, no mannequin, no person

DO NOT:
- Change garment design, pattern, color, proportions, or length
- Add shadows, gradients, reflections, or glow effects
- Zoom in or enlarge any part of the garment
- Alter fabric texture or sheen beyond natural studio lighting
""".strip()


def _prepare_for_gpt(image_bytes: bytes) -> bytes:
    """
    Resizes the BRIA-cleaned PNG to 1024px max before sending to GPT.
    - Reduces input token cost significantly
    - GPT-image-1.5 doesn't benefit from inputs larger than 1024px
    - Keeps PNG format to preserve transparency
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    w, h = img.size

    if max(w, h) > 1024:
        scale = 1024 / max(w, h)
        img = img.resize(
            (int(w * scale), int(h * scale)),
            Image.LANCZOS,
        )

    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True, compress_level=6)
    return out.getvalue()


def apply_gpt_premium_3d(cleaned_png_bytes: bytes) -> bytes:
    """
    Sends the BRIA-cleaned PNG to GPT-image-1.5 for 3D reconstruction.

    Settings chosen for cost/quality balance:
    - model: gpt-image-1.5 (20% cheaper than gpt-image-1, better editing)
    - quality: medium (~$0.034 base, ~$0.025 with smaller input)
    - size: 1024x1024 (Cloudinary upscales to 2000x2000 anyway)

    Input is the already-cleaned transparent PNG from your BRIA pipeline,
    not the raw photo — this reduces input token cost and gives GPT
    a simpler task (add 3D form) rather than a complex one (remove bg + 3D).
    """
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY is missing in .env")

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Resize to 1024px to reduce input tokens
    prepared_bytes = _prepare_for_gpt(cleaned_png_bytes)

    response = client.images.edit(
        model="gpt-image-1.5",      # 20% cheaper than gpt-image-1
        image=("product.png", io.BytesIO(prepared_bytes), "image/png"),
        prompt=_GPT_PROMPT,
        size="1024x1024",
        quality="medium",           # $0.034 vs $0.133 for high
        output_format="jpeg",       # JPEG output = fewer output tokens than PNG
        output_compression=88,      # 0-100; 88 is visually lossless for e-commerce
    )

    # GPT returns base64 by default for edits
    image_data = response.data[0]

    if hasattr(image_data, "url") and image_data.url:
        result = requests.get(image_data.url, timeout=60)
        if result.status_code != 200:
            raise Exception(f"Failed to download GPT image: {result.status_code}")
        result_bytes = result.content
    elif hasattr(image_data, "b64_json") and image_data.b64_json:
        result_bytes = base64.b64decode(image_data.b64_json)
    else:
        raise Exception("GPT-image-1.5 returned no image data")

    # Output is JPEG (RGB) — convert to RGBA PNG for the rest of the pipeline.
    # Cloudinary handles the final white background compositing anyway.
    img = Image.open(io.BytesIO(result_bytes)).convert("RGBA")
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True, compress_level=6)
    return out.getvalue()


# ----------------------------
# Input validation
# ----------------------------

def _validate_input(image_bytes: bytes) -> Image.Image:
    """
    Validates and normalizes the input image.
    - Rejects corrupt/invalid files
    - Applies EXIF rotation (critical for phone photos)
    - Rejects oversized images
    Raises ValueError with a clear message on failure.

    Note: We skip img.verify() because WhatsApp and some phone cameras
    produce JPEGs with non-standard metadata that causes verify() to
    raise false positives. Instead we do a full decode attempt which
    is a more reliable real-world validity check.
    """
    if len(image_bytes) < 1024:
        raise ValueError("File too small to be a valid image.")

    try:
        # Full decode is more reliable than verify() for real-world images
        img = Image.open(io.BytesIO(image_bytes))
        img.load()  # Force full decode — catches truly corrupt files
    except Exception:
        raise ValueError("Invalid or corrupt image file.")

    # Fix EXIF rotation — phone/WhatsApp photos are often rotated
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass  # Non-critical — continue without rotation fix

    mp = (img.width * img.height) / 1_000_000
    if mp > MAX_INPUT_MP:
        raise ValueError(f"Image too large ({mp:.1f}MP). Max is {MAX_INPUT_MP}MP.")

    return img


# ----------------------------
# BRIA helpers
# ----------------------------

def _require_api_key() -> str:
    if not BRIA_API_KEY:
        raise Exception("BRIA_API_KEY is missing in .env")
    return BRIA_API_KEY


def _download_image_bytes(image_url: str) -> bytes:
    response = requests.get(image_url, timeout=180)
    if response.status_code != 200:
        raise Exception(f"Failed to download BRIA result: {response.status_code}")
    return response.content


def _extract_image_url(data: dict) -> Optional[str]:
    return data.get("result", {}).get("image_url")


def _poll_status_url(status_url: str) -> bytes:
    api_key = _require_api_key()
    headers = {"api_token": api_key}
    start = time.time()

    while True:
        if time.time() - start > POLL_TIMEOUT_SECONDS:
            raise Exception("BRIA async request timed out")

        response = requests.get(status_url, headers=headers, timeout=60)

        if response.status_code != 200:
            raise Exception(
                f"BRIA status check failed: {response.status_code} {response.text}"
            )

        data = response.json()
        status = str(data.get("status", "")).lower()

        if status in {"completed", "succeeded", "success", "done"}:
            image_url = _extract_image_url(data)
            if not image_url:
                raise Exception(f"BRIA completed without image_url: {data}")
            return _download_image_bytes(image_url)

        if status in {"failed", "error", "cancelled", "canceled"}:
            raise Exception(f"BRIA async request failed: {data}")

        time.sleep(POLL_INTERVAL_SECONDS)


def _post_bria_json(url: str, image_bytes: bytes, sync: bool = True) -> bytes:
    api_key = _require_api_key()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "api_token": api_key,
    }

    payload = {
        "image": image_b64,
        "sync": sync,
        "preserve_alpha": True,
        "visual_input_content_moderation": False,
        "visual_output_content_moderation": False,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=180)

    # sync route
    if sync:
        if response.status_code != 200:
            raise Exception(
                f"BRIA sync request failed: {response.status_code} {response.text}"
            )
        data = response.json()
        image_url = _extract_image_url(data)
        if not image_url:
            raise Exception(f"BRIA returned no image_url: {data}")
        return _download_image_bytes(image_url)

    # async route
    if response.status_code != 202:
        raise Exception(
            f"BRIA async request failed: {response.status_code} {response.text}"
        )

    data = response.json()
    status_url = data.get("status_url")
    request_id = data.get("request_id")

    if not status_url and request_id:
        status_url = f"{BRIA_BASE_URL}/v2/status/{request_id}"

    if not status_url:
        raise Exception(
            f"BRIA async request returned no status_url/request_id: {data}"
        )

    return _poll_status_url(status_url)


def remove_background(image_bytes: bytes) -> bytes:
    """
    Calls BRIA remove_background and returns the RGBA PNG.
    Avoids unnecessary re-encoding if BRIA already returns RGBA.
    """
    result_bytes = _post_bria_json(REMOVE_BG_URL, image_bytes, sync=True)

    img = Image.open(io.BytesIO(result_bytes))
    if img.mode != "RGBA":
        img = img.convert("RGBA")
        out = io.BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()

    # Return original bytes untouched — no lossy round-trip
    return result_bytes


def apply_bria_postprocess(image_bytes: bytes, mode: BriaMode = "none") -> bytes:
    if mode == "none":
        return image_bytes

    if mode == "enhance":
        result_bytes = _post_bria_json(ENHANCE_URL, image_bytes, sync=False)
    elif mode == "increase_resolution":
        result_bytes = _post_bria_json(
            INCREASE_RESOLUTION_URL, image_bytes, sync=False
        )
    else:
        raise Exception(f"Unsupported BRIA mode: {mode}")

    img = Image.open(io.BytesIO(result_bytes)).convert("RGBA")
    out = io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


# ----------------------------
# Mask refinement helpers
# ----------------------------

def _read_png_rgba(png_bytes: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(png_bytes)).convert("RGBA"))


def _read_rgb(image_bytes: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))


def _png_bytes_from_rgba(arr: np.ndarray) -> bytes:
    out = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="RGBA").save(out, format="PNG")
    return out.getvalue()


def _mask_u8(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8) * 255


def _bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, 0, 0
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1, y1


def _adaptive_alpha_thresholds(alpha: np.ndarray) -> Tuple[int, int]:
    nz = alpha[alpha > 0]
    if nz.size < 64:
        return 20, 96

    soft_q = float(np.quantile(nz, 0.08))
    strong_q = float(np.quantile(nz, 0.60))

    soft_thr = int(max(12, min(40, round(soft_q))))
    strong_thr = int(
        max(64, min(180, round(max(strong_q, soft_thr + 24))))
    )

    return soft_thr, strong_thr


def _best_component(
    mask_u8: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Selects the best connected component from a binary mask.

    Scoring improvements vs original:
    - Reduced top_penalty weight (0.40 → 0.15): long garments naturally
      have centroids in the lower half; penalising this was incorrect.
    - Increased tall_bonus weight (0.15 → 0.45): tall components should
      be strongly preferred — they are almost always the main garment.
    """
    h, w = mask_u8.shape[:2]
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    if n <= 1:
        return np.zeros_like(mask_u8), {"label": -1, "score": -1.0}

    best_label = -1
    best_score = -1e18

    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area <= 0:
            continue

        cx, cy = centroids[i]
        area_ratio = area / float(h * w)
        centrality_penalty = abs(cx - (w / 2.0)) / max(1.0, w / 2.0)
        top_penalty = cy / max(1.0, h)
        tall_bonus = hh / max(1.0, h)

        # FIX: top_penalty reduced (was 0.40), tall_bonus increased (was 0.15)
        # Long garment centroids sit low — punishing this was selecting wrong components
        score = (
            (6.0 * area_ratio)
            - (0.85 * centrality_penalty)
            - (0.15 * top_penalty)   # was 0.40
            + (0.45 * tall_bonus)    # was 0.15
        )

        if score > best_score:
            best_score = float(score)
            best_label = i

    out = np.zeros_like(mask_u8)
    if best_label > 0:
        out[labels == best_label] = 255

    return out, {"label": int(best_label), "score": float(best_score)}


def _distance_core(mask_u8: np.ndarray) -> np.ndarray:
    if mask_u8.max() == 0:
        return np.zeros_like(mask_u8)

    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return np.zeros_like(mask_u8)

    thr = max(2.0, 0.32 * float(dist.max()))
    core = np.where(dist >= thr, 255, 0).astype(np.uint8)

    if core.max() == 0:
        thr = max(1.0, 0.20 * float(dist.max()))
        core = np.where(dist >= thr, 255, 0).astype(np.uint8)

    return core


def _make_grow_kernel(mask: np.ndarray) -> np.ndarray:
    """
    Creates an image-relative grow kernel so that geodesic growth
    is proportional to image size rather than fixed at 5px.

    A 400px image and a 4000px image previously both got a 5px kernel,
    which is a 10x difference in proportional effect.
    """
    h, w = mask.shape[:2]
    k = max(3, min(9, round(min(h, w) * 0.008)))
    if k % 2 == 0:
        k += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _geodesic_grow(
    seed_u8: np.ndarray,
    barrier_u8: np.ndarray,
    kernel: np.ndarray,
    max_iters: int = 64,
) -> np.ndarray:
    grown = seed_u8.copy()
    barrier_u8 = _mask_u8(barrier_u8)

    for _ in range(max_iters):
        dil = cv2.dilate(grown, kernel, iterations=1)
        new_grown = cv2.bitwise_and(dil, barrier_u8)
        if np.array_equal(new_grown, grown):
            break
        grown = new_grown

    return grown


def _bridge_break_reconstruct(
    mask_u8: np.ndarray, seed_u8: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Any]]:
    h, w = mask_u8.shape[:2]
    k = int(max(3, min(7, round(min(h, w) * 0.006))))
    if k % 2 == 0:
        k += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    eroded = cv2.erode(mask_u8, kernel, iterations=1)

    if eroded.max() == 0:
        return mask_u8.copy(), {"bridge_kernel": k, "used_eroded": False}

    if seed_u8.max() > 0:
        seed_on_eroded = cv2.bitwise_and(eroded, seed_u8)
        if seed_on_eroded.max() == 0:
            seed_on_eroded, _ = _best_component(eroded)
    else:
        seed_on_eroded, _ = _best_component(eroded)

    if seed_on_eroded.max() == 0:
        return mask_u8.copy(), {"bridge_kernel": k, "used_eroded": False}

    rec = _geodesic_grow(seed_on_eroded, mask_u8, kernel, max_iters=96)
    return rec, {"bridge_kernel": k, "used_eroded": True}


def _row_widths(
    mask_u8: np.ndarray,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    x0, y0, x1, y1 = _bbox_from_mask(mask_u8)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0,), dtype=np.float32), (0, 0, 0, 0)

    roi = mask_u8[y0:y1, x0:x1] > 0
    widths = roi.sum(axis=1).astype(np.float32)
    return widths, (x0, y0, x1, y1)


def _detect_garment_type(mask_u8: np.ndarray, hint: str = "auto") -> str:
    if hint in {"top", "long"}:
        return hint

    widths, (x0, y0, x1, y1) = _row_widths(mask_u8)
    if widths.size == 0:
        return "top"

    h = max(1, y1 - y0)
    w = max(1, x1 - x0)
    bbox_h_over_w = h / float(w)

    nz = widths[widths > 0]
    maxw = float(nz.max()) if nz.size else 1.0
    bottom_band = (
        nz[max(0, int(0.80 * len(nz))) :]
        if nz.size
        else nz
    )
    mid_band = (
        nz[
            max(0, int(0.40 * len(nz))) : max(1, int(0.60 * len(nz)))
        ]
        if nz.size
        else nz
    )

    bottom_ratio = (
        float(np.median(bottom_band) / maxw) if bottom_band.size else 0.0
    )
    mid_ratio = float(np.median(mid_band) / maxw) if mid_band.size else 0.0

    long_like = (
        (bbox_h_over_w > 1.25)
        or (bottom_ratio > 0.48)
        or (bottom_ratio >= mid_ratio)
    )
    return "long" if long_like else "top"


def _find_tail_cut_row(
    mask_u8: np.ndarray, garment_type: str
) -> Optional[Dict[str, Any]]:
    widths, (x0, y0, x1, y1) = _row_widths(mask_u8)
    if widths.size < 20:
        return None

    n = widths.size
    win = int(max(5, min(19, round(n * 0.03))))
    if win % 2 == 0:
        win += 1

    kernel = np.ones((win,), dtype=np.float32) / float(win)
    sm = np.convolve(widths, kernel, mode="same")
    maxw = float(max(1.0, sm.max()))

    start_frac = 0.38 if garment_type == "top" else 0.52
    drop_ratio = 0.38 if garment_type == "top" else 0.26
    tail_ratio = 0.22 if garment_type == "top" else 0.14

    start = int(n * start_frac)
    needed = 4
    best_r = None

    for r in range(start, n - needed - 3):
        prev_peak = float(sm[max(0, r - 12) : r].max()) if r > 0 else maxw
        tail_med = float(np.median(sm[r + 1 : r + 1 + 18]))
        cond = (sm[r] < drop_ratio * max(prev_peak, 1.0)) and (
            tail_med < tail_ratio * maxw
        )

        if not cond:
            continue

        ok = True
        for rr in range(r, min(n, r + needed)):
            if sm[rr] >= tail_ratio * maxw * 1.25:
                ok = False
                break

        if ok:
            best_r = r
            break

    if best_r is None:
        return None

    cut_row = y0 + best_r
    tail = np.zeros_like(mask_u8)
    tail[cut_row + 1 :, :] = mask_u8[cut_row + 1 :, :]
    tail = _mask_u8(tail)

    if tail.max() == 0:
        return None

    h, w = mask_u8.shape[:2]
    n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(
        tail, connectivity=8
    )
    kept = np.zeros_like(tail)

    for i in range(1, n2):
        x, y, ww, hh, area = stats2[i]
        touches_bottom = (y + hh) >= (h - 1)
        clearly_below = y >= (cut_row + 2)
        if touches_bottom or clearly_below:
            kept[labels2 == i] = 255

    if kept.max() == 0:
        return None

    tx0, ty0, tx1, ty1 = _bbox_from_mask(kept)
    tail_w = max(1, tx1 - tx0)
    tail_h = max(1, ty1 - ty0)
    tail_area = int((kept > 0).sum())
    total_area = int((mask_u8 > 0).sum())

    row_at_cut = int(widths[best_r])
    tail_widths = (
        widths[best_r + 1 :]
        if (best_r + 1) < n
        else np.zeros((0,), dtype=np.float32)
    )
    tail_med_width = float(np.median(tail_widths)) if tail_widths.size else 0.0

    geom = {
        "cut_row": int(cut_row),
        "bridge_width_px": int(row_at_cut),
        "bridge_width_frac": float(row_at_cut / maxw),
        "tail_med_width_frac": float(tail_med_width / maxw),
        "tail_area_frac": float(tail_area / max(total_area, 1)),
        "tail_aspect": float(tail_h / max(1, tail_w)),
    }

    return {"tail_mask": kept, "geom": geom}


def _rgb_region_features(
    rgb: np.ndarray, region_u8: np.ndarray
) -> Dict[str, float]:
    sel = region_u8 > 0
    if not np.any(sel):
        return {
            "lab_L": 0.0,
            "lab_a": 0.0,
            "lab_b": 0.0,
            "sat_mean": 0.0,
            "grad_mean": 0.0,
            "skin_frac": 0.0,
        }

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    L = lab[:, :, 0][sel].astype(np.float32)
    a = lab[:, :, 1][sel].astype(np.float32)
    b = lab[:, :, 2][sel].astype(np.float32)
    sat = hsv[:, :, 1][sel].astype(np.float32)
    grad_vals = grad[sel].astype(np.float32)

    Y = ycrcb[:, :, 0]
    Cr = ycrcb[:, :, 1]
    Cb = ycrcb[:, :, 2]
    skin = (
        (Cr >= 135) & (Cr <= 180) & (Cb >= 85) & (Cb <= 135) & (Y > 40)
    )
    skin_frac = float(skin[sel].mean()) if np.any(sel) else 0.0

    return {
        "lab_L": float(L.mean()),
        "lab_a": float(a.mean()),
        "lab_b": float(b.mean()),
        "sat_mean": float(sat.mean()),
        "grad_mean": float(grad_vals.mean()),
        "skin_frac": float(skin_frac),
    }


def _tail_should_be_removed(
    rgb: np.ndarray,
    core_u8: np.ndarray,
    tail_u8: np.ndarray,
    geom: Dict[str, float],
    garment_type: str,
) -> Tuple[bool, Dict[str, float]]:
    core_f = _rgb_region_features(rgb, core_u8)
    tail_f = _rgb_region_features(rgb, tail_u8)

    d_lab = (
        (core_f["lab_L"] - tail_f["lab_L"]) ** 2
        + (core_f["lab_a"] - tail_f["lab_a"]) ** 2
        + (core_f["lab_b"] - tail_f["lab_b"]) ** 2
    ) ** 0.5

    grad_ratio = tail_f["grad_mean"] / max(core_f["grad_mean"], 1e-6)
    sat_ratio = tail_f["sat_mean"] / max(core_f["sat_mean"], 1e-6)

    geom_positive = (
        (
            geom["bridge_width_frac"]
            < (0.16 if garment_type == "top" else 0.14)
        )
        and (
            geom["tail_med_width_frac"]
            < (0.18 if garment_type == "top" else 0.16)
        )
        and (
            geom["tail_aspect"] > (1.4 if garment_type == "top" else 1.6)
        )
        and (
            geom["tail_area_frac"]
            < (0.16 if garment_type == "top" else 0.10)
        )
    )

    vote = 0
    if d_lab > (14.0 if garment_type == "top" else 22.0):
        vote += 1
    if grad_ratio < (0.82 if garment_type == "top" else 0.70):
        vote += 1
    if (
        sat_ratio < (0.80 if garment_type == "top" else 0.65)
        or sat_ratio > 1.75
    ):
        vote += 1
    if tail_f["skin_frac"] > (0.06 if garment_type == "top" else 0.12):
        vote += 2

    remove = bool(
        geom_positive and vote >= (1 if garment_type == "top" else 2)
    )

    info = {
        "d_lab": float(d_lab),
        "grad_ratio": float(grad_ratio),
        "sat_ratio": float(sat_ratio),
        "tail_skin_frac": float(tail_f["skin_frac"]),
        "remove_vote": int(vote),
        "geom_positive": bool(geom_positive),
    }

    return remove, info


def _detect_dark_legs(
    rgb: np.ndarray,
    mask_u8: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """
    Detects dark legging/trouser regions at the bottom of the mask.

    These appear in mannequin/hanger shots where the model wears black
    leggings under a light garment. They survive tail detection because
    they attach broadly at the hem — no narrow bridge to break.

    Two-zone strategy:
    - SCAN zone (bottom 30%): used to compute dark_frac and decide
      whether leggings are present at all. Smaller zone = higher
      dark_frac signal because we're not diluting with garment fabric.
    - REMOVE zone (bottom 25%): the actual pixels to zero out.
      Slightly smaller than the scan zone so we never clip the hem.

    Keeping scan and remove zones separate means we can detect
    confidently while being conservative about what we delete.
    """
    x0, y0, x1, y1 = _bbox_from_mask(mask_u8)
    if x1 <= x0 or y1 <= y0:
        return None

    bbox_h = y1 - y0

    # SCAN zone: bottom 30% of bounding box
    scan_top = y0 + int(bbox_h * 0.70)
    # REMOVE zone: bottom 25% (conservative — stops before the hem)
    remove_top = y0 + int(bbox_h * 0.75)

    scan_mask = mask_u8[scan_top:y1, x0:x1].copy()
    scan_rgb = rgb[scan_top:y1, x0:x1]

    if scan_mask.max() == 0 or scan_rgb.size == 0:
        return None

    # LAB L-channel: brightness independent of hue
    lab = cv2.cvtColor(scan_rgb, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0]  # 0-255 in OpenCV LAB

    # L < 60 captures black and dark-grey leggings
    dark_pixels_scan = ((L < 60) & (scan_mask > 0)).astype(np.uint8) * 255

    dark_area = int(dark_pixels_scan.sum() / 255)
    scan_mask_area = int(scan_mask.sum() / 255)

    if scan_mask_area == 0:
        return None

    dark_frac = dark_area / float(scan_mask_area)

    # Threshold lowered to 0.15: the tighter scan zone means even
    # partially-visible leggings give a strong signal now
    if dark_frac < 0.15:
        return None

    # Build the REMOVE mask — slightly higher than scan_top to protect hem
    # Expand dark pixels in the remove zone using the full scan detection
    remove_rgb = rgb[remove_top:y1, x0:x1]
    remove_mask = mask_u8[remove_top:y1, x0:x1].copy()

    if remove_rgb.size > 0 and remove_mask.max() > 0:
        lab_remove = cv2.cvtColor(remove_rgb, cv2.COLOR_RGB2LAB)
        L_remove = lab_remove[:, :, 0]
        dark_pixels_remove = (
            (L_remove < 60) & (remove_mask > 0)
        ).astype(np.uint8) * 255
    else:
        dark_pixels_remove = np.zeros(
            (y1 - remove_top, x1 - x0), dtype=np.uint8
        )

    # Reconstruct in full image coordinates
    dark_full = np.zeros_like(mask_u8)
    dark_full[remove_top:y1, x0:x1] = dark_pixels_remove

    return {
        "dark_leg_mask": dark_full,
        "dark_frac": float(dark_frac),
        "scan_top": int(scan_top),
        "remove_top": int(remove_top),
    }


def _defringe_alpha(
    alpha: np.ndarray, final_u8: np.ndarray, erosion_px: int = 2
) -> np.ndarray:
    """
    Removes the 1-2px halo ring that appears when compositing on white.

    How it works:
    1. Erode the hard mask slightly to cut the fringe pixel ring.
    2. Apply a narrow Gaussian feather only in the transition zone
       (the ring between eroded and original mask boundary).
    3. Interior pixels are untouched — no blurring of the product.

    This eliminates halos without softening the garment itself.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (erosion_px * 2 + 1, erosion_px * 2 + 1),
    )
    eroded = cv2.erode(final_u8, kernel, iterations=1)

    # Feather only the narrow transition ring
    blurred = cv2.GaussianBlur(
        eroded.astype(np.float32), (5, 5), sigmaX=1.0
    )

    # Transition zone = pixels that were in the mask but got eroded off
    transition = (final_u8 > 0) & (eroded == 0)

    result = eroded.astype(np.float32)
    result[transition] = blurred[transition]

    return np.clip(result, 0, 255).astype(np.uint8)


# ----------------------------
# Main mask refinement
# ----------------------------

def refine_cutout_mask_safe(
    cutout_png_bytes: bytes,
    original_rgb_bytes: bytes,
    garment_hint: str = "auto",
) -> Tuple[bytes, Dict[str, Any]]:
    rgba = _read_png_rgba(cutout_png_bytes)
    rgb = _read_rgb(original_rgb_bytes)
    alpha = rgba[:, :, 3]

    soft_thr, strong_thr = _adaptive_alpha_thresholds(alpha)
    soft_u8 = _mask_u8(alpha > soft_thr)
    strong_u8 = _mask_u8(alpha > strong_thr)

    seed_u8, seed_meta = _best_component(strong_u8)
    if seed_u8.max() == 0:
        dist_core = _distance_core(soft_u8)
        seed_u8, seed_meta = _best_component(dist_core)

    if seed_u8.max() == 0:
        return cutout_png_bytes, {
            "fallback": "no_seed",
            "soft_thr": soft_thr,
            "strong_thr": strong_thr,
        }

    # Use image-relative kernel (FIX: was fixed 5x5 regardless of image size)
    kernel = _make_grow_kernel(soft_u8)
    grown_u8 = _geodesic_grow(seed_u8, soft_u8, kernel, max_iters=96)

    rec_u8, rec_meta = _bridge_break_reconstruct(grown_u8, seed_u8)
    garment_type = _detect_garment_type(rec_u8, hint=garment_hint)

    cut = _find_tail_cut_row(rec_u8, garment_type=garment_type)
    remove_tail_u8 = np.zeros_like(rec_u8)
    tail_reason = {}

    if cut is not None:
        core_above_u8 = rec_u8.copy()
        core_above_u8[cut["geom"]["cut_row"] + 1 :, :] = 0

        remove, tail_reason = _tail_should_be_removed(
            rgb=rgb,
            core_u8=core_above_u8,
            tail_u8=cut["tail_mask"],
            geom=cut["geom"],
            garment_type=garment_type,
        )

        if remove:
            remove_tail_u8 = cut["tail_mask"]

        if garment_type == "top":
            kernel_top = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (3, 3)
            )
            remove_tail_u8 = cv2.dilate(
                remove_tail_u8, kernel_top, iterations=1
            )

    final_u8 = rec_u8.copy()
    final_u8[remove_tail_u8 > 0] = 0

    # --- Dark leg removal (mannequin/hanger shots with black leggings) ---
    # Runs after tail removal so we're working on the already-cleaned mask.
    # Only removes dark pixels if the main garment is light — this prevents
    # mistakenly cutting pixels from dark jackets, black dresses, etc.
    dark_leg_result = _detect_dark_legs(rgb, final_u8)
    dark_legs_removed = False
    garment_mean_L = 0.0

    if dark_leg_result is not None:
        lab_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        garment_mean_L = float(lab_full[:, :, 0][final_u8 > 0].mean())

        # Light garment threshold: mean L > 100
        # Dark garments (black jackets, dark dresses) have mean L < 100
        if garment_mean_L > 100:
            final_u8[dark_leg_result["dark_leg_mask"] > 0] = 0
            dark_legs_removed = True

    # Safety fallback — prevent over-aggressive trimming
    before_area = int((grown_u8 > 0).sum())
    after_area = int((final_u8 > 0).sum())
    area_ratio = after_area / float(max(before_area, 1))

    bx0, by0, bx1, by1 = _bbox_from_mask(grown_u8)
    fx0, fy0, fx1, fy1 = _bbox_from_mask(final_u8)

    grown_h = max(1, by1 - by0)
    final_h = max(1, fy1 - fy0)
    bbox_h_ratio = final_h / float(grown_h)

    fallback_reason = None
    if area_ratio < 0.84:
        fallback_reason = "area_drop"
    elif bbox_h_ratio < 0.92 and garment_type == "long":
        fallback_reason = "bbox_height_drop_long"

    if fallback_reason:
        final_u8 = rec_u8.copy()

    # Apply defringe to clean up halo pixels before compositing
    refined_alpha = _defringe_alpha(alpha, final_u8)

    out_rgba = rgba.copy()
    out_rgba[:, :, 3] = np.where(
        final_u8 > 0, refined_alpha, 0
    ).astype(np.uint8)

    stats = {
        "soft_thr": int(soft_thr),
        "strong_thr": int(strong_thr),
        "seed_meta": seed_meta,
        "bridge_meta": rec_meta,
        "garment_type": garment_type,
        "tail_cut_found": bool(cut is not None),
        "tail_geom": cut["geom"] if cut is not None else None,
        "tail_reason": tail_reason,
        "removed_tail_pixels": int((remove_tail_u8 > 0).sum()),
        "before_area": before_area,
        "after_area": after_area,
        "area_ratio": float(area_ratio),
        "bbox_h_ratio": float(bbox_h_ratio),
        "fallback_reason": fallback_reason,
        "dark_legs_removed": dark_legs_removed,
        "dark_leg_frac": float(dark_leg_result["dark_frac"]) if dark_leg_result else 0.0,
        "garment_mean_L": float(garment_mean_L),
    }

    return _png_bytes_from_rgba(out_rgba), stats


# ----------------------------
# Main pipeline
# ----------------------------

def process_product_image(
    image_bytes: bytes,
    mode: BriaMode = "none",
    garment_hint: str = "auto",
) -> bytes:
    """
    Full pipeline:
    1. Validate + normalize input (EXIF fix, size check, format check)
    2. Re-serialize as JPEG for BRIA (smaller payload, faster upload)
    3. Remove background via BRIA
    4. Refine mask (geodesic grow, bridge break, tail removal, defringe)
    5. Optional post-processing:
       - increase_resolution: BRIA upscale
       - enhance: BRIA enhance
       - premium_3d: GPT-image-1.5 medium quality 3D reconstruction
    6. Compress and return final PNG

    premium_3d mode cost: ~$0.025–0.028 per image
    (gpt-image-1.5, medium quality, 1024px input from BRIA-cleaned PNG)
    """

    # Step 1: Validate and normalize
    validated_img = _validate_input(image_bytes)

    # Step 2: Re-serialize as JPEG for BRIA
    # Sends a smaller payload — BRIA doesn't need PNG transparency at this stage
    bria_buf = io.BytesIO()
    validated_img.convert("RGB").save(bria_buf, format="JPEG", quality=92)
    bria_input_bytes = bria_buf.getvalue()

    # Also keep a normalized RGB version for mask feature analysis
    rgb_buf = io.BytesIO()
    validated_img.convert("RGB").save(rgb_buf, format="PNG")
    normalized_rgb_bytes = rgb_buf.getvalue()

    # Step 3: Remove background
    cutout_png = remove_background(bria_input_bytes)

    # Step 4: Refine mask
    refined_png, stats = refine_cutout_mask_safe(
        cutout_png_bytes=cutout_png,
        original_rgb_bytes=normalized_rgb_bytes,
        garment_hint=garment_hint,
    )

    logger.debug("Mask refinement stats: %s", stats)

    # Step 5: Post-processing
    if mode == "increase_resolution":
        logger.info("POST: increase_resolution")
        refined_png = apply_bria_postprocess(
            refined_png, mode="increase_resolution"
        )

    elif mode == "enhance":
        logger.info("POST: enhance")
        refined_png = apply_bria_postprocess(refined_png, mode="enhance")

    elif mode == "premium_3d":
        # Send BRIA-cleaned PNG to GPT-image-1.5 for 3D reconstruction.
        # Using cleaned PNG (not raw photo) reduces input tokens = lower cost.
        # GPT output goes to Cloudinary next (in the API layer) for final
        # 2000x2000 white background + centering — same as all other modes.
        logger.info("POST: premium_3d via GPT-image-1.5")
        try:
            refined_png = apply_gpt_premium_3d(refined_png)
        except Exception as e:
            # If GPT fails, log it and fall back to the clean BRIA output
            # rather than returning an error — vendor still gets a usable image
            logger.error("GPT premium_3d failed, falling back to BRIA output: %s", e)

    # Step 6: Compress final PNG
    # compress_level=6: ~80% of level-9 savings at ~20% of the CPU cost
    final_img = Image.open(io.BytesIO(refined_png)).convert("RGBA")
    out = io.BytesIO()
    final_img.save(out, format="PNG", optimize=True, compress_level=6)

    return out.getvalue()