"""
batch_3d.py — OpenAI Batch API integration for premium_3d mode.

WHY BATCH:
  OpenAI Batch API gives exactly 50% off all token rates.
  Your current cost: ~$0.05/image (real-time)
  Batch cost:        ~$0.025/image (async, processed within 24h)

HOW IT WORKS:
  1. Vendor uploads images → BRIA pipeline runs immediately (fast, cheap)
  2. Cleaned PNGs are submitted as a batch job to OpenAI
  3. OpenAI processes within 24h (usually much faster, ~1-4h)
  4. Your server polls or receives webhook → results saved
  5. Vendor gets notified → downloads finished catalog images

FLOW:
  submit_batch_job(list of cleaned PNGs)
    → uploads each image as a File
    → creates a .jsonl batch file
    → submits to OpenAI Batch API
    → returns batch_id

  poll_batch_job(batch_id)
    → checks status
    → if complete, downloads all results
    → returns dict of custom_id → image bytes

  get_batch_status(batch_id)
    → returns status + progress counts
"""

import base64
import io
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI
from PIL import Image

logger = logging.getLogger(__name__)

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


def _get_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY is missing in .env")
    return OpenAI(api_key=OPENAI_API_KEY)


def _prepare_for_gpt(image_bytes: bytes) -> bytes:
    """Resize to 1024px max to reduce input tokens."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    w, h = img.size
    if max(w, h) > 1024:
        scale = 1024 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True, compress_level=6)
    return out.getvalue()


def _upload_image_file(client: OpenAI, image_bytes: bytes, filename: str) -> str:
    """
    Uploads a PNG to OpenAI Files API for use in batch requests.
    Returns the file_id.
    """
    prepared = _prepare_for_gpt(image_bytes)
    file_tuple = (filename, io.BytesIO(prepared), "image/png")

    response = client.files.create(
        file=file_tuple,
        purpose="vision",
    )
    return response.id


def submit_batch_job(
    images: List[Dict[str, Any]],
) -> str:
    """
    Submits a batch of cleaned PNGs to OpenAI Batch API.

    Args:
        images: List of dicts with keys:
            - custom_id (str): your internal ID, e.g. image UUID
            - png_bytes (bytes): BRIA-cleaned PNG

    Returns:
        batch_id (str): use this to poll status and retrieve results

    Cost: ~50% of real-time pricing = ~$0.025/image at medium quality
    Turnaround: within 24h (usually 1-4h in practice)
    """
    client = _get_client()

    if not images:
        raise ValueError("No images provided to batch job")

    if len(images) > 100:
        raise ValueError(
            f"Batch size {len(images)} exceeds limit of 100. "
            "Split into multiple batches."
        )

    logger.info("Uploading %d images to OpenAI Files API...", len(images))

    # Step 1: Upload each image as a File and collect file_ids
    file_ids: Dict[str, str] = {}
    for item in images:
        custom_id = item["custom_id"]
        png_bytes = item["png_bytes"]
        filename = f"{custom_id}.png"

        try:
            file_id = _upload_image_file(client, png_bytes, filename)
            file_ids[custom_id] = file_id
            logger.debug("Uploaded %s → file_id=%s", custom_id, file_id)
        except Exception as e:
            logger.error("Failed to upload image %s: %s", custom_id, e)
            raise

    # Step 2: Build the .jsonl batch request file
    # Each line is one image edit request
    jsonl_lines = []
    for item in images:
        custom_id = item["custom_id"]
        file_id = file_ids[custom_id]

        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/images/edits",
            "body": {
                "model": "gpt-image-1.5",
                "prompt": _GPT_PROMPT,
                "size": "1024x1024",
                "quality": "medium",
                "output_format": "jpeg",
                "output_compression": 88,
                "image": file_id,  # reference uploaded file by ID
                "n": 1,
            },
        }
        jsonl_lines.append(json.dumps(request))

    jsonl_content = "\n".join(jsonl_lines).encode("utf-8")

    # Step 3: Upload the .jsonl batch file
    logger.info("Uploading batch .jsonl file (%d requests)...", len(jsonl_lines))
    batch_file = client.files.create(
        file=("batch_requests.jsonl", io.BytesIO(jsonl_content), "application/jsonl"),
        purpose="batch",
    )

    # Step 4: Submit the batch job
    logger.info("Submitting batch job to OpenAI...")
    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/images/edits",
        completion_window="24h",
        metadata={
            "description": f"Zipa premium_3d batch — {len(images)} images",
        },
    )

    logger.info(
        "Batch submitted. batch_id=%s status=%s", batch.id, batch.status
    )

    return batch.id


def get_batch_status(batch_id: str) -> Dict[str, Any]:
    """
    Returns current batch status and progress counts.

    Possible statuses:
      validating  — OpenAI is checking the batch file
      in_progress — processing
      finalizing  — wrapping up
      completed   — all done, results available
      failed      — batch failed entirely
      expired     — exceeded 24h window
      cancelled   — manually cancelled

    Returns dict with:
      status, total, completed, failed, batch_id
    """
    client = _get_client()
    batch = client.batches.retrieve(batch_id)

    counts = batch.request_counts
    return {
        "batch_id": batch_id,
        "status": batch.status,
        "total": counts.total if counts else 0,
        "completed": counts.completed if counts else 0,
        "failed": counts.failed if counts else 0,
        "output_file_id": batch.output_file_id,
        "error_file_id": batch.error_file_id,
        "created_at": batch.created_at,
        "completed_at": getattr(batch, "completed_at", None),
    }


def retrieve_batch_results(batch_id: str) -> Dict[str, bytes]:
    """
    Downloads and returns all completed results from a finished batch.

    Args:
        batch_id: the batch_id from submit_batch_job()

    Returns:
        Dict mapping custom_id → RGBA PNG bytes
        (ready to send to Cloudinary)

    Raises:
        Exception if batch is not yet completed or has failed
    """
    client = _get_client()
    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        raise Exception(
            f"Batch {batch_id} is not completed yet. "
            f"Current status: {batch.status}"
        )

    if not batch.output_file_id:
        raise Exception(f"Batch {batch_id} completed but has no output file")

    # Download the results .jsonl
    logger.info("Downloading batch results for batch_id=%s", batch_id)
    result_content = client.files.content(batch.output_file_id)
    result_text = result_content.read().decode("utf-8")

    results: Dict[str, bytes] = {}
    errors: Dict[str, str] = {}

    for line in result_text.strip().split("\n"):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse result line: %s", e)
            continue

        custom_id = record.get("custom_id", "unknown")
        error = record.get("error")

        if error:
            logger.error(
                "Result error for %s: %s", custom_id, error
            )
            errors[custom_id] = str(error)
            continue

        response = record.get("response", {})
        body = response.get("body", {})
        data = body.get("data", [])

        if not data:
            logger.error("No image data in result for %s", custom_id)
            errors[custom_id] = "no image data in response"
            continue

        image_record = data[0]
        b64 = image_record.get("b64_json")
        url = image_record.get("url")

        try:
            if b64:
                image_bytes = base64.b64decode(b64)
            elif url:
                resp = requests.get(url, timeout=60)
                if resp.status_code != 200:
                    raise Exception(f"HTTP {resp.status_code}")
                image_bytes = resp.content
            else:
                raise Exception("no b64_json or url in response")

            # Convert to RGBA PNG for Cloudinary pipeline
            img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
            out = io.BytesIO()
            img.save(out, format="PNG", optimize=True, compress_level=6)
            results[custom_id] = out.getvalue()

        except Exception as e:
            logger.error("Failed to decode image for %s: %s", custom_id, e)
            errors[custom_id] = str(e)

    if errors:
        logger.warning(
            "Batch %s completed with %d errors: %s",
            batch_id, len(errors), errors
        )

    logger.info(
        "Batch %s: %d successful, %d errors",
        batch_id, len(results), len(errors)
    )

    return results


def poll_until_complete(
    batch_id: str,
    poll_interval_seconds: int = 60,
    max_wait_seconds: int = 86400,  # 24h
) -> Dict[str, bytes]:
    """
    Blocking poll — waits until batch completes then returns results.
    Use this for server-side background jobs, not for request handlers.

    For production, use get_batch_status() in a scheduled task instead
    of blocking a thread for hours.
    """
    start = time.time()

    while True:
        if time.time() - start > max_wait_seconds:
            raise Exception(
                f"Batch {batch_id} did not complete within "
                f"{max_wait_seconds}s"
            )

        status_info = get_batch_status(batch_id)
        status = status_info["status"]

        logger.info(
            "Batch %s status=%s completed=%d/%d",
            batch_id,
            status,
            status_info["completed"],
            status_info["total"],
        )

        if status == "completed":
            return retrieve_batch_results(batch_id)

        if status in {"failed", "expired", "cancelled"}:
            raise Exception(
                f"Batch {batch_id} ended with status: {status}"
            )

        time.sleep(poll_interval_seconds)