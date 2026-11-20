#!/bin/bash
# LTX-2.5 serverless worker startup.
#
# Downloads the ~45GB int8 model set from HuggingFace on first boot
# (comfyicu/LTX-2.5 mirror - the official Lightricks/LTX-2.5 repo is gated
# and returns 401 without an accepted licence + token; the mirror files are
# byte-identical for the transformer and VAEs, and the text encoder differs
# only by a 2.4KB safetensors header).
#
# Models go to /comfyui/models by default. If a RunPod network volume is
# attached at /runpod-volume, models are stored there instead
# (/runpod-volume/comfyui/models) and shared across workers.
#
# IMPORTANT: do NOT trust file size alone for "download complete". aria2
# pre-allocates the destination to full size immediately, so a stalled
# partial looks complete by size. Integrity is enforced by aria2
# --checksum=sha-256; size is only a cheap secondary guard.

echo "=========================================="
echo "Starting ComfyUI LTX-2.5 Video Worker"
echo "=========================================="

# Resolve the models directory (volume preferred, local disk fallback).
if [ -d /runpod-volume ] && [ -w /runpod-volume ]; then
    MODELS_DIR="/runpod-volume/comfyui/models"
    mkdir -p "$MODELS_DIR"
    # Point ComfyUI's models dir at the volume.
    rm -rf /comfyui/models
    ln -s "$MODELS_DIR" /comfyui/models
    echo "Using network volume for models: $MODELS_DIR"
else
    MODELS_DIR="${MODELS_DIR:-/comfyui/models}"
    echo "Using local disk for models: $MODELS_DIR"
fi

PER_FILE_BUDGET="${PER_FILE_BUDGET:-1500}"   # hard per-file wall-clock cap (seconds)
SPEED_FLOOR="${SPEED_FLOOR:-1M}"              # abort+retry a source that trickles below this

# Optional HF token (needed only if sources are ever switched to a gated repo).
ARIA2_AUTH=()
if [ -n "$HF_TOKEN" ]; then
    ARIA2_AUTH=(--header="Authorization: Bearer $HF_TOKEN")
fi

# basename | dest-subdir | size(bytes) | sha256 | huggingface resolve path
# Sizes + SHA-256 from the comfyicu/LTX-2.5 HuggingFace tree API.
read -r -d '' MODELS <<'EOF'
ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors|diffusion_models|21504034224|c4279eeff115cbeaca494bd2183e7d768c38fe85a184dc6afbb7159157c44334|comfyicu/LTX-2.5/resolve/main/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors
gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors|text_encoders|15372971786|09a89e084de1a149c3de60cfe9dfd3e5161967eb09eea39e806fcdeffdd568de|comfyicu/LTX-2.5/resolve/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors
gemma4_e2b_it_int8_convrot.safetensors|text_encoders|5199997904|efeca0fcad2f863e5ed0a75e3af952b72bc963604c1dda6d20aee87a32b17566|Comfy-Org/gemma-4/resolve/main/text_encoders/gemma4_e2b_it_int8_convrot.safetensors
ltx-2.5-video-vae-bf16.safetensors|vae|1472223346|847e14ca7f3355debca0cea4eaa24ac0fbcdf0061da054ac89ca638a869ddba3|comfyicu/LTX-2.5/resolve/main/vae/ltx-2.5-video-vae-bf16.safetensors
ltx-2.5-audio-vae-bf16.safetensors|vae|364866540|c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5|comfyicu/LTX-2.5/resolve/main/vae/ltx-2.5-audio-vae-bf16.safetensors
ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors|latent_upscale_models|995778752|eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8|comfyicu/LTX-2.5/resolve/main/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
EOF

download_model() {
    local base="$1" subdir="$2" size="$3" sha="$4" hf="$5"
    local dir="${MODELS_DIR}/${subdir}"
    local dest="${dir}/${base}"
    mkdir -p "$dir"

    # Already present at exact size? Fresh downloads are checksum-validated,
    # so an exact-size existing file is a safe skip (helps warm disk /
    # re-entrant boots and volume reuse across workers).
    if [ -f "$dest" ] && [ "$(stat -c%s "$dest" 2>/dev/null || echo 0)" = "$size" ]; then
        echo "  ✓ present: $base"
        return 0
    fi
    rm -f "$dest" "$dest.aria2"

    local src="https://huggingface.co/${hf}"
    local attempt=1
    while [ "$attempt" -le 3 ]; do
        echo "  ↓ ${base} from HF (attempt ${attempt})"
        if timeout "$PER_FILE_BUDGET" aria2c \
                -x16 -s16 -k1M -c \
                --checksum="sha-256=${sha}" \
                --lowest-speed-limit="$SPEED_FLOOR" \
                --max-tries=2 --retry-wait=5 --timeout=60 \
                --console-log-level=warn --summary-interval=0 \
                "${ARIA2_AUTH[@]}" \
                -d "$dir" -o "$base" "$src" \
           && [ "$(stat -c%s "$dest" 2>/dev/null || echo 0)" = "$size" ]; then
            echo "  ✓ ${base} (sha256 verified)"
            return 0
        fi
        echo "  ⚠ ${base} failed/invalid (attempt ${attempt}); cleaning + retrying"
        rm -f "$dest" "$dest.aria2"
        attempt=$((attempt + 1))
        sleep $((attempt * 5))
    done
    echo "  ✗ FAILED: ${base}"
    return 1
}

echo ""
echo "=========================================="
echo "Downloading models from HuggingFace (~45GB)..."
echo "=========================================="
fail=0
while IFS='|' read -r base subdir size sha hf; do
    [ -z "$base" ] && continue
    download_model "$base" "$subdir" "$size" "$sha" "$hf" || fail=1
done < <(printf '%s\n' "$MODELS")

echo ""
echo "=========================================="
echo "Verifying models (fail-hard before ComfyUI)..."
echo "=========================================="
missing=0
while IFS='|' read -r base subdir size sha hf; do
    [ -z "$base" ] && continue
    dest="${MODELS_DIR}/${subdir}/${base}"
    actual=$(stat -c%s "$dest" 2>/dev/null || echo 0)
    if [ "$actual" = "$size" ]; then
        echo "✓ $base"
    else
        echo "✗ MISSING/incomplete: $base ($actual/$size)"
        missing=1
    fi
done < <(printf '%s\n' "$MODELS")

if [ "$fail" = "1" ] || [ "$missing" = "1" ]; then
    echo "FATAL: model set incomplete — exiting so RunPod replaces this worker (never serve half-initialized)."
    exit 1
fi

echo ""
echo "=========================================="
echo "Starting ComfyUI..."
echo "=========================================="
cd /comfyui
python main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch &

echo "Waiting for ComfyUI..."
sleep 15

for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
        echo "✓ ComfyUI is running!"
        break
    fi
    echo "  Waiting... ($i/30)"
    sleep 2
done

echo ""
echo "=========================================="
echo "Starting Handler..."
echo "=========================================="
cd /
python -u /handler.py
