import asyncio
import io
import logging
from typing import List, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.model import process_product_image
from app.batch_3d import (
    submit_batch_job,
    get_batch_status,
    retrieve_batch_results,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Zipa BRIA API")

MAX_UPLOAD_MB = 20
MAX_BATCH_SIZE = 100

# Timeouts per mode
MODE_TIMEOUTS = {
    "none": 240.0,
    "enhance": 300.0,
    "increase_resolution": 300.0,
    "premium_3d": 360.0,  # BRIA (~30s) + GPT (~30s) + buffer
}


@app.get("/")
def root():
    return {"status": "BRIA API server running"}


# ----------------------------
# Standard single-image endpoint (real-time)
# ----------------------------

@app.post("/remove-bg")
async def remove_bg(
    file: UploadFile = File(...),
    mode: Literal["none", "enhance", "increase_resolution", "premium_3d"] = Form("none"),
    garment_hint: Literal["auto", "top", "long"] = Form("auto"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed.")

    input_bytes = await file.read()

    if not input_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    if len(input_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum upload size is {MAX_UPLOAD_MB}MB.",
        )

    timeout = MODE_TIMEOUTS.get(mode, 240.0)

    try:
        output_bytes = await asyncio.wait_for(
            run_in_threadpool(process_product_image, input_bytes, mode, garment_hint),
            timeout=timeout,
        )

    except asyncio.TimeoutError:
        logger.error("Processing timed out (mode=%s) for file: %s", mode, file.filename)
        raise HTTPException(status_code=504, detail="Image processing timed out.")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except HTTPException:
        raise

    except Exception:
        logger.exception("Processing failed (mode=%s) for file: %s", mode, file.filename)
        raise HTTPException(status_code=500, detail="Image processing failed.")

    return StreamingResponse(
        io.BytesIO(output_bytes),
        media_type="image/png",
        headers={
            "Content-Disposition": f'inline; filename="{file.filename or "output"}.png"'
        },
    )


# ----------------------------
# Batch endpoints (50% cheaper, async)
# ----------------------------

@app.post("/batch/submit")
async def batch_submit(
    files: List[UploadFile] = File(...),
    garment_hint: Literal["auto", "top", "long"] = Form("auto"),
):
    """
    Submits multiple images for premium_3d batch processing.

    Cost: ~$0.025/image (50% off real-time via OpenAI Batch API)
    Turnaround: within 24h (typically 1-4h)

    Flow:
    1. Each image goes through BRIA background removal immediately
    2. Cleaned PNGs are submitted as a batch to OpenAI
    3. Returns batch_id — poll /batch/status/{batch_id} for progress
    4. When complete, fetch results from /batch/results/{batch_id}

    Returns: { batch_id, submitted_count, image_ids }
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum batch size is {MAX_BATCH_SIZE}.",
        )

    # Validate all files first before processing any
    for f in files:
        if not f.content_type or not f.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"File '{f.filename}' is not an image.",
            )

    import uuid

    batch_inputs = []
    bria_errors = []

    # Step 1: Run BRIA background removal on all images (real-time, fast)
    # We do this synchronously per image — BRIA is the fast part (~5-10s each)
    for f in files:
        raw_bytes = await f.read()

        if not raw_bytes:
            bria_errors.append(f.filename)
            continue

        if len(raw_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
            bria_errors.append(f"{f.filename} (too large)")
            continue

        try:
            # Run BRIA only (mode="none") to get the clean PNG
            cleaned_png = await asyncio.wait_for(
                run_in_threadpool(process_product_image, raw_bytes, "none", garment_hint),
                timeout=240.0,
            )

            batch_inputs.append({
                "custom_id": str(uuid.uuid4()),
                "filename": f.filename,
                "png_bytes": cleaned_png,
            })

        except Exception as e:
            logger.error("BRIA failed for %s: %s", f.filename, e)
            bria_errors.append(f.filename)

    if not batch_inputs:
        raise HTTPException(
            status_code=500,
            detail="All images failed BRIA processing. No batch submitted.",
        )

    # Step 2: Submit cleaned PNGs to OpenAI Batch API
    try:
        batch_id = await run_in_threadpool(submit_batch_job, batch_inputs)
    except Exception as e:
        logger.exception("Batch submission failed")
        raise HTTPException(
            status_code=500,
            detail=f"Batch submission failed: {str(e)}",
        )

    return {
        "batch_id": batch_id,
        "submitted_count": len(batch_inputs),
        "skipped_count": len(bria_errors),
        "skipped_files": bria_errors,
        "image_ids": [item["custom_id"] for item in batch_inputs],
        "estimated_cost_usd": round(len(batch_inputs) * 0.025, 4),
        "message": (
            f"Batch submitted. {len(batch_inputs)} images queued for 3D processing. "
            "Poll /batch/status/{batch_id} for updates. "
            "Typically completes within 1-4 hours."
        ),
    }


@app.get("/batch/status/{batch_id}")
async def batch_status(batch_id: str):
    """
    Returns current status and progress of a batch job.

    Statuses:
      validating  — OpenAI is validating the request file
      in_progress — processing
      finalizing  — wrapping up outputs
      completed   — done, fetch results from /batch/results/{batch_id}
      failed      — batch failed
      expired     — exceeded 24h window
      cancelled   — manually cancelled
    """
    try:
        status = await run_in_threadpool(get_batch_status, batch_id)
    except Exception as e:
        logger.exception("Failed to get batch status for %s", batch_id)
        raise HTTPException(status_code=500, detail=str(e))

    return status


class BatchResultRequest(BaseModel):
    cloudinary_upload: bool = True  # if True, upload each result to Cloudinary


@app.get("/batch/results/{batch_id}")
async def batch_results(batch_id: str):
    """
    Downloads completed batch results and returns Cloudinary URLs.

    Only call this when /batch/status returns status="completed".
    Returns: { results: [{ custom_id, result_url }], count }
    """
    # Check status first
    try:
        status_info = await run_in_threadpool(get_batch_status, batch_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if status_info["status"] != "completed":
        raise HTTPException(
            status_code=202,
            detail=(
                f"Batch not ready yet. "
                f"Status: {status_info['status']}. "
                f"Progress: {status_info['completed']}/{status_info['total']}"
            ),
        )

    # Download all results
    try:
        result_images = await run_in_threadpool(retrieve_batch_results, batch_id)
    except Exception as e:
        logger.exception("Failed to retrieve batch results for %s", batch_id)
        raise HTTPException(status_code=500, detail=str(e))

    if not result_images:
        raise HTTPException(
            status_code=500,
            detail="Batch completed but no results could be retrieved.",
        )

    # Upload each result to Cloudinary
    # Reuse the same upload logic from apply-background route
    import cloudinary
    import cloudinary.uploader

    cloudinary.config(
        cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
        api_key=os.environ.get("CLOUDINARY_API_KEY"),
        api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
    )

    output_results = []

    for custom_id, png_bytes in result_images.items():
        try:
            # Compress before upload
            img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True, compress_level=6)
            compressed = buf.getvalue()

            uploaded = cloudinary.uploader.upload(
                compressed,
                folder="imageforge/batch_3d",
                resource_type="image",
                format="png",
            )

            result_url = cloudinary.CloudinaryImage(uploaded["public_id"]).build_url(
                secure=True,
                transformation=[
                    {"width": 1700, "height": 1700, "crop": "fit", "gravity": "center"},
                    {"width": 2000, "height": 2000, "crop": "pad", "background": "white", "gravity": "center"},
                    {"format": "png"},
                ],
            )

            output_results.append({
                "custom_id": custom_id,
                "result_url": result_url,
                "public_id": uploaded["public_id"],
            })

        except Exception as e:
            logger.error("Cloudinary upload failed for %s: %s", custom_id, e)
            output_results.append({
                "custom_id": custom_id,
                "result_url": None,
                "error": str(e),
            })

    return {
        "batch_id": batch_id,
        "count": len(output_results),
        "results": output_results,
    }


import os
from PIL import Image