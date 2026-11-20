"""
ComfyUI LTX-2.5 Video Serverless Handler for RunPod.

Two modes (same worker, two workflows converted from the official ComfyUI
v0.46.1 blueprints "Text to Video (LTX-2.5)" / "Image to Video (LTX-2.5)"):
  1. t2v: text-to-video (no image input)
  2. i2v: image-to-video (image_base64 given, used as the first frame)

Both modes generate video WITH synchronized audio (LTX-2.5 is an
audio-video model). The distilled transformer runs a fixed calibrated
schedule (3 first-pass + 8 refinement steps); steps/CFG are intentionally
not exposed.

Model files (auto-downloaded by start.sh into /comfyui/models):
  diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors
  text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors
  text_encoders/gemma4_e2b_it_int8_convrot.safetensors  (prompt enhancer;
      required for /prompt validation even with enhancement OFF)
  vae/ltx-2.5-video-vae-bf16.safetensors
  vae/ltx-2.5-audio-vae-bf16.safetensors
  latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors

Input format:
{
    "input": {
        "prompt": str,              # Required
        "image_base64": str,        # Optional - when present, I2V mode
        "negative_prompt": str,     # Optional (default: blueprint default)
        "width": int,               # Optional (default 1280, snapped to /32)
        "height": int,              # Optional (default 720, snapped to /32)
        "duration": int,            # Optional seconds (default 5, max 5)
        "frames": int,              # Optional - overrides duration
                                    # (frames = duration*fps + 1)
        "fps": int,                 # Optional (default 24)
        "seed": int,                # Optional (random if not set)
        "i2v_strength_first": float,  # I2V only (default 0.7)
        "i2v_strength_second": float, # I2V only (default 1.0)
        "img_compression": int,     # I2V only (default 18)
        "timeout": int,             # Optional seconds (default 840)
    }
}

Output format:
{
    "ok": true,
    "video": "<base64-encoded mp4>",
    "seed": int,
    "mode": "t2v" | "i2v",
    "width": int, "height": int, "duration": int, "fps": int,
    "frame_count": int,
    "timings": {"elapsed_s": float},
    "worker": {"id": ..., "hostname": ...},
}

Failures return {"ok": false, "error_code": ..., "error_message": ...,
"retryable": bool, ...}.

Node-ID maps (must stay in sync with workflow_t2v.json / workflow_i2v.json):
  T2V (subgraph 455):
    455:452  PrimitiveStringMultiline  .value       prompt
    455:433  CLIPTextEncode            .text        negative prompt
    455:421  RandomNoise               .noise_seed  seed (first pass)
    455:429  RandomNoise               .noise_seed  seed (refinement)
    455:450  PrimitiveInt              .value       duration (seconds)
    455:451  PrimitiveInt              .value       width
    455:448  PrimitiveInt              .value       height
    455:449  PrimitiveInt              .value       fps
  I2V (subgraph 452):
    1        LoadImage                 .image       input_image.png
    452:409  PrimitiveStringMultiline  .value       prompt
    452:419  CLIPTextEncode            .text        negative prompt
    452:423  RandomNoise               .noise_seed  seed
    452:442  RandomNoise               .noise_seed  seed
    452:413  PrimitiveInt              .value       duration (seconds)
    452:418  PrimitiveInt              .value       width
    452:421  PrimitiveInt              .value       height
    452:425  PrimitiveInt              .value       fps
    452:429  LTXVImgToVideoInplace     .strength    i2v first pass
    452:439  LTXVImgToVideoInplace     .strength    i2v refinement
    452:436  LTXVPreprocess            .img_compression
"""

import base64
import json
import logging
import os
import random
import socket
import sys
import time
import traceback
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

import runpod

# ---------------------------------------------------------------- logging

logger = logging.getLogger("LTX25-Handler")
logger.setLevel(logging.INFO)
_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                      datefmt="%Y-%m-%d %H:%M:%S")
)
logger.handlers = []
logger.addHandler(_console)

# ---------------------------------------------------------------- config

COMFYUI_HOST = "127.0.0.1"
COMFYUI_PORT = 8188
COMFYUI_URL = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"
COMFYUI_INPUT_DIR = "/comfyui/input"
COMFYUI_OUTPUT_DIR = "/comfyui/output"

DEFAULT_NEGATIVE_PROMPT = (
    "pc game, console game, video game, cartoon, childish, ugly"
)

# 10s clips can OOM the VAE decode on 24GB; cap at 5s for now.
MAX_DURATION_S = 5
MIN_DIMENSION = 256
MAX_DIMENSION = 1280
# Snap to multiples of 16: preserves standard dims (720, 1280, 1024, 768,
# 512, 480). NOTE: /32 would break 720p (720 is not a multiple of 32 and
# would snap down to 704); the official template itself uses 1280x720.
DIMENSION_STEP = 16

DEFAULTS = {
    "width": 1280,
    "height": 720,
    "duration": 5,
    "fps": 24,
    "seed": None,
    "timeout": 840,
    "i2v_strength_first": 0.7,
    "i2v_strength_second": 1.0,
    "img_compression": 18,
}

# node-id maps: workflow key -> {param: [(node_id, input_key), ...]}
T2V_MAP = {
    "prompt": [("455:452", "value")],
    "negative_prompt": [("455:433", "text")],
    "seed": [("455:421", "noise_seed"), ("455:429", "noise_seed")],
    "duration": [("455:450", "value")],
    "width": [("455:451", "value")],
    "height": [("455:448", "value")],
    "fps": [("455:449", "value")],
}

I2V_MAP = {
    "prompt": [("452:409", "value")],
    "negative_prompt": [("452:419", "text")],
    "seed": [("452:423", "noise_seed"), ("452:442", "noise_seed")],
    "duration": [("452:413", "value")],
    "width": [("452:418", "value")],
    "height": [("452:421", "value")],
    "fps": [("452:425", "value")],
    "i2v_strength_first": [("452:429", "strength")],
    "i2v_strength_second": [("452:439", "strength")],
    "img_compression": [("452:436", "img_compression")],
}


def _worker_metadata() -> Dict[str, Any]:
    return {
        "id": (
            os.getenv("RUNPOD_WORKER_ID")
            or os.getenv("RUNPOD_POD_ID")
            or os.getenv("RUNPOD_MACHINE_ID")
        ),
        "hostname": socket.gethostname(),
    }


def _failure_response(
    *,
    error_code: str,
    error_message: str,
    retryable: bool,
    elapsed_s: float,
) -> Dict[str, Any]:
    return {
        "ok": False,
        "error": error_message,
        "error_code": error_code,
        "error_message": error_message,
        "retryable": retryable,
        "worker": _worker_metadata(),
        "timings": {"elapsed_s": elapsed_s},
    }


def _classify_exception(exc: Exception) -> tuple:
    """Return (error_code, retryable)."""
    message = str(exc).lower()
    if isinstance(exc, TimeoutError):
        return ("WORKFLOW_TIMEOUT", True)
    # NB: HTTPError subclasses URLError, so check it FIRST.
    if isinstance(exc, urllib.error.HTTPError):
        if "prompt" in message:
            return ("WORKFLOW_QUEUE_FAILED", True)
        return ("COMFYUI_HTTP_ERROR", True)
    if isinstance(exc, urllib.error.URLError):
        return ("COMFYUI_UNREACHABLE", True)
    if "workflow error" in message:
        return ("WORKFLOW_EXECUTION_ERROR", False)
    if "comfyui /prompt" in message:
        return ("WORKFLOW_QUEUE_FAILED", True)
    if "http error" in message and "prompt" in message:
        return ("WORKFLOW_QUEUE_FAILED", True)
    if "incorrect padding" in message or "invalid base64" in message:
        return ("INVALID_INPUT", False)
    if "missing required field" in message:
        return ("INVALID_INPUT", False)
    return ("INTERNAL_ERROR", True)


# ---------------------------------------------------------------- helpers

def wait_for_comfyui(timeout: int = 180) -> bool:
    logger.info(f"Waiting for ComfyUI server at {COMFYUI_URL}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(f"{COMFYUI_URL}/system_stats",
                                          timeout=5)
            if resp.status == 200:
                logger.info("ComfyUI server is ready!")
                return True
        except Exception as e:
            logger.debug(f"Waiting... ({e})")
        time.sleep(2)
    logger.error(f"ComfyUI server did not start within {timeout}s")
    return False


def save_input_image(image_data: str,
                     filename: str = "input_image.png") -> str:
    if "base64," in image_data:
        image_data = image_data.split("base64,")[1]
    image_bytes = base64.b64decode(image_data)
    filepath = os.path.join(COMFYUI_INPUT_DIR, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    logger.info(f"Saved input image to {filepath} "
                f"({os.path.getsize(filepath)} bytes)")
    return filepath


def load_workflow(mode: str) -> Dict:
    path = f"/workflow_{mode}.json"
    logger.info(f"Loading workflow from {path}...")
    with open(path) as f:
        return json.load(f)


def _set_input(workflow: Dict, node_id: str, key: str, value) -> None:
    if node_id in workflow:
        workflow[node_id]["inputs"][key] = value
    else:
        logger.warning(f"workflow node {node_id} missing; "
                       f"cannot set {key}={value!r}")


def apply_params(workflow: Dict, node_map: Dict, params: Dict) -> None:
    for param, targets in node_map.items():
        if param not in params:
            continue
        for node_id, key in targets:
            _set_input(workflow, node_id, key, params[param])


def queue_prompt(workflow: Dict) -> str:
    data = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:
        # ComfyUI returns {"error": ..., "node_errors": ...} on 400; surface
        # it so a validation failure names the offending node instead of a
        # bare "HTTP Error 400".
        body = e.read().decode("utf-8", errors="replace")[:3000]
        logger.error(f"ComfyUI /prompt rejected ({e.code}): {body}")
        raise RuntimeError(f"ComfyUI /prompt HTTP {e.code}: {body}") from e
    result = json.loads(resp.read().decode("utf-8"))
    prompt_id = result.get("prompt_id")
    logger.info(f"Prompt queued with ID: {prompt_id}")
    return prompt_id


def wait_for_completion(prompt_id: str, timeout: int = 840) -> Dict:
    logger.info(f"Waiting for completion (timeout: {timeout}s)...")
    start = time.time()
    last_log = 0
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(
                f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
            history = json.loads(resp.read().decode("utf-8"))
            if prompt_id in history:
                status = history[prompt_id].get("status", {})
                if status.get("status_str") == "error":
                    msgs = status.get("messages", [["Unknown error"]])
                    raise RuntimeError(f"Workflow error: {msgs[0]}")
                outputs = history[prompt_id].get("outputs", {})
                if outputs:
                    logger.info(f"Generation completed in "
                                f"{time.time() - start:.1f}s")
                    return outputs
        except RuntimeError:
            raise
        except Exception as e:
            logger.debug(f"Status check error: {e}")
        elapsed = int(time.time() - start)
        if elapsed - last_log >= 30:
            logger.info(f"Progress: {elapsed}s elapsed...")
            last_log = elapsed
        time.sleep(2)
    raise TimeoutError(f"Generation timed out after {timeout} seconds")


def get_output_video(outputs: Dict) -> Optional[str]:
    """Return base64 MP4 from ComfyUI history outputs (or newest .mp4)."""
    candidates = []
    for node_id, node_output in outputs.items():
        for key in ["videos", "video", "gifs", "images", "files"]:
            if key not in node_output:
                continue
            items = node_output[key]
            if not isinstance(items, list):
                items = [items]
            for item in items:
                if isinstance(item, dict):
                    filename = item.get("filename")
                    subfolder = item.get("subfolder", "")
                elif isinstance(item, str):
                    filename = item
                    subfolder = ""
                else:
                    continue
                if not filename:
                    continue
                path = os.path.join(COMFYUI_OUTPUT_DIR, subfolder, filename) \
                    if subfolder else os.path.join(COMFYUI_OUTPUT_DIR, filename)
                if os.path.exists(path):
                    candidates.append(path)
    if not candidates and os.path.exists(COMFYUI_OUTPUT_DIR):
        for root, _dirs, files in os.walk(COMFYUI_OUTPUT_DIR):
            for name in sorted(files, reverse=True):
                if name.endswith((".mp4", ".webm", ".mov")):
                    candidates.append(os.path.join(root, name))
                    break
    if not candidates:
        logger.warning("No output video found in results")
        return None
    # Prefer mp4 files; newest first.
    candidates.sort(key=lambda p: (not p.endswith(".mp4"),
                                   -os.path.getmtime(p)))
    path = candidates[0]
    size = os.path.getsize(path)
    logger.info(f"Found output video: {path} ({size} bytes)")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    try:
        os.remove(path)  # keep the worker disk clean between jobs
    except OSError:
        pass
    return data


def _snap_dimension(value: int) -> int:
    value = max(MIN_DIMENSION, min(MAX_DIMENSION, int(value)))
    return max(MIN_DIMENSION,
               (value // DIMENSION_STEP) * DIMENSION_STEP)


# ---------------------------------------------------------------- handler

def handler(job: Dict) -> Dict:
    job_id = job.get("id", "unknown")
    start_time = time.time()
    logger.info(f"=== LTX-2.5 VIDEO JOB: {job_id} ===")
    try:
        job_input = job.get("input", {})
        prompt = job_input.get("prompt")
        if not prompt:
            return _failure_response(
                error_code="INVALID_INPUT",
                error_message="Missing required field: prompt",
                retryable=False,
                elapsed_s=round(time.time() - start_time, 3),
            )

        image_b64 = job_input.get("image_base64") or job_input.get("image")
        mode = "i2v" if image_b64 else "t2v"
        logger.info(f"Mode: {mode.upper()}")

        width = _snap_dimension(job_input.get("width", DEFAULTS["width"]))
        height = _snap_dimension(job_input.get("height", DEFAULTS["height"]))
        fps = max(12, min(30, int(job_input.get("fps", DEFAULTS["fps"]))))
        duration = int(job_input.get("duration", DEFAULTS["duration"]))
        if job_input.get("frames") is not None:
            frames = int(job_input["frames"])
            duration = max(1, round((frames - 1) / fps))
        duration = max(1, min(MAX_DURATION_S, duration))
        frame_count = duration * fps + 1

        seed = job_input.get("seed")
        if seed is None or seed == -1:
            seed = random.randint(0, 2**31 - 1)
            logger.info(f"Generated random seed: {seed}")

        params = {
            "prompt": prompt,
            "negative_prompt": job_input.get(
                "negative_prompt", DEFAULT_NEGATIVE_PROMPT),
            "width": width,
            "height": height,
            "duration": duration,
            "fps": fps,
            "seed": seed,
        }
        if mode == "i2v":
            params["i2v_strength_first"] = float(job_input.get(
                "i2v_strength_first", DEFAULTS["i2v_strength_first"]))
            params["i2v_strength_second"] = float(job_input.get(
                "i2v_strength_second", DEFAULTS["i2v_strength_second"]))
            params["img_compression"] = int(job_input.get(
                "img_compression", DEFAULTS["img_compression"]))

        timeout = int(job_input.get("timeout", DEFAULTS["timeout"]))
        logger.info(f"Params: {width}x{height}, {duration}s, {fps}fps, "
                    f"{frame_count} frames, seed={seed}")

        if not wait_for_comfyui():
            return _failure_response(
                error_code="COMFYUI_BOOT_TIMEOUT",
                error_message="ComfyUI server not available",
                retryable=True,
                elapsed_s=round(time.time() - start_time, 3),
            )

        if mode == "i2v":
            save_input_image(image_b64)

        workflow = load_workflow(mode)
        apply_params(workflow, T2V_MAP if mode == "t2v" else I2V_MAP, params)

        prompt_id = queue_prompt(workflow)
        outputs = wait_for_completion(prompt_id, timeout)
        video_data = get_output_video(outputs)

        if not video_data:
            return _failure_response(
                error_code="NO_OUTPUT_VIDEO",
                error_message="No video output generated",
                retryable=False,
                elapsed_s=round(time.time() - start_time, 3),
            )

        elapsed = round(time.time() - start_time, 3)
        logger.info(f"=== JOB COMPLETED in {elapsed}s ===")
        return {
            "ok": True,
            "video": video_data,
            "seed": seed,
            "mode": mode,
            "width": width,
            "height": height,
            "duration": duration,
            "fps": fps,
            "frame_count": frame_count,
            "worker": _worker_metadata(),
            "timings": {"elapsed_s": elapsed},
        }

    except Exception as e:
        elapsed = round(time.time() - start_time, 3)
        logger.error(f"Job failed after {elapsed}s: {e}")
        logger.error(traceback.format_exc())
        error_code, retryable = _classify_exception(e)
        return _failure_response(
            error_code=error_code,
            error_message=str(e),
            retryable=retryable,
            elapsed_s=elapsed,
        )


if __name__ == "__main__":
    logger.info("LTX-2.5 video serverless handler starting")
    logger.info(f"ComfyUI URL: {COMFYUI_URL}")
    runpod.serverless.start({"handler": handler})
