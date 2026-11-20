# RunPod Serverless worker: LTX-2.5 text/image-to-video via ComfyUI.
#
# Architecture: ComfyUI v0.46.1 (Comfy-Org) + official LTX-2.5 blueprints
# converted to API format (workflow_t2v.json / workflow_i2v.json). The
# handler talks to ComfyUI over HTTP (127.0.0.1:8188), like the
# runpod/worker-comfyui pattern, but built from scratch because there is no
# runpod/worker-comfyui LTX image.
#
# All nodes used by the LTX-2.5 blueprints are in ComfyUI core as of
# v0.46.1 (LTXV* nodes, TextGenerateLTX2Prompt, ResizeImageMaskNode,
# ComfySwitchNode, ...), so NO custom nodes are installed.
#
# Models (~43GB) are NOT baked into the image: start.sh downloads them
# from HuggingFace on first boot into /comfyui/models (or
# /runpod-volume/comfyui/models when a network volume is attached).
# Image size: ~9-10GB. Cold start: model download ~10-20 min on first boot
# (initTimeout 3600s); later jobs on a warm worker load from local disk.
#
# GPU: RTX 4090 (24GB) with the FP8 merged transformer (Eros10 + DMD +
# IC-LoRAs baked in) + uncensored int8 text encoder.
# 5s clips verified to fit; 10s clips may OOM at VAE decode, so the
# handler caps duration at 5s.
#
# Build (CI): GitHub Actions workflow in .github/workflows/build.yml
# pushes ghcr.io/<owner>/ltx25-worker:latest.

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    git \
    wget \
    curl \
    aria2 \
    unzip \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
RUN python3.12 -m ensurepip --upgrade \
    && python3.12 -m pip install --no-cache-dir --upgrade pip \
    && ln -sf /usr/local/bin/pip3.12 /usr/local/bin/pip \
    && ln -sf /usr/local/bin/pip3.12 /usr/local/bin/pip3

# PyTorch (CUDA 12.8). Installed before ComfyUI's requirements.txt so pip
# sees torch/torchvision as already satisfied and does not pull the
# default-index builds.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# ComfyUI v0.46.1 (Comfy-Org; the comfyanonymous repo moved here).
# v0.46.1 ships the official LTX-2.5 blueprints and all LTXV nodes in core.
RUN git clone --branch v0.46.1 --depth 1 https://github.com/Comfy-Org/ComfyUI.git /comfyui
WORKDIR /comfyui
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir runpod huggingface_hub

RUN mkdir -p models/diffusion_models \
    models/text_encoders \
    models/vae \
    models/latent_upscale_models \
    models/loras \
    input \
    output \
    workflows

COPY handler.py /handler.py
COPY start.sh /start.sh
COPY workflow_t2v.json /workflow_t2v.json
COPY workflow_i2v.json /workflow_i2v.json
RUN chmod +x /start.sh

# Build-time sanity check: torch + runpod import, workflows parse as JSON.
RUN python3 -c "\
import torch, runpod, json; \
json.load(open('/workflow_t2v.json')); \
json.load(open('/workflow_i2v.json')); \
print(f'PyTorch {torch.__version__}, runpod {runpod.__version__}'); \
print('workflows parse OK'); \
"

WORKDIR /
EXPOSE 8188
CMD ["/start.sh"]
