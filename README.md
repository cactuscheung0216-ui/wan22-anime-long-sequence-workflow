# Wan 2.2 I2V · 24-Round Multi-Round Video Pipeline (V109 / V92)

**Long-sequence multi-round Wan 2.2 I2V-A14B generation + commercial 60 fps 4-stage post-processing.**

- **Latest Release — V109 (`544×960 → 960×1680 @ 60 fps`, 115.70 s)**: Full-body 9:16 pipeline restoring the **P1 golden sampling kernel** (`steps 2+4+4=10`, `MLC=1`, `lock_start_slots=1`, `add_reference_latents=false`, `lr=0.80`, `motion=1.18`, per-round `LAB 0.90` color match), full-body anti-crop framing guardrails, and a **bundled 4-stage post-processing runner** (`postprocess_24R_V109.py`) outputting commercial BT.709 H.264 High@L4.2 60 fps (`6,942` frames, `17.73 Mbps`, `+faststart`).
- **Legacy Baseline — V92 (`832×1232 @ 30 fps`, 61.73 s)**: Original 2:3 anime head-to-thigh choreography release (`1,852` frames @ 30 fps), kept alongside V109 for reference and comparison.

> 本仓库提供 **Wan 2.2 I2V-A14B** 多轮长序列连续视频生成（24 轮潜空间链式续写，一镜到底）及配套工业级后处理完整管线。  
> 最新版本 **V109** 重点解决了四个核心工程问题：  
> 1. **采样核回归修复**：退回实验期引入重影与背景网格的参数，恢复经过 24 轮全量验证的 P1 黄金采样核；  
> 2. **全身防裁切构图锁**：支持 9:16 全身入画（`544×960 @ 16fps`，单轮 `5.06s`，24 轮 `115.70s`），解决仅上半身挥手、脚部出画与负向护栏冲突；  
> 3. **完整四阶段后处理闭环**：仓库直接附带 [`postprocess_24R_V109.py`](postprocess_24R_V109.py)（接缝直方图校准 → 2× 模型超分 + 轻量 CAS 锐化 → 可选局部 Detailer → `FL_RIFE` 4× 光流补帧至 64fps + BT.709 `60fps` 商用编码）；  
> 4. **RIFE 分片重叠吞帧修复与色调保护**：修复 `RIFE_OVERLAP_TRIM` 吞帧漏洞，默认关闭老旧灰蒙渐变调色曲线，保留全长 24 轮尾声。

---

## Table of Contents

- [Release Tracks (V109 vs V92)](#release-tracks-v109-vs-v92)
- [V109 Architecture & Fixes](#v109-architecture--fixes)
- [Bundled Files](#bundled-files)
- [Requirements](#requirements)
- [Custom Nodes](#custom-nodes)
- [Models](#models)
- [Quick Start (V109)](#quick-start-v109)
- [Post-Processing Pipeline (`postprocess_24R_V109.py`)](#post-processing-pipeline-postprocess_24r_v109py)
- [Workflow Graph & Parameters](#workflow-graph--parameters)
- [Measured Results](#measured-results)
- [Changelog](#changelog)
- [License](#license)
- [Credits](#credits)

---

## Release Tracks (V109 vs V92)

| Track | Generation Res / FPS | Frames (24R, drop=4) | Delivery Res / FPS | Duration | Post-Process Script | Status |
|---|---|---|---|---|---|---|
| **V109 (Recommended)** | `544×960` @ `16 fps` | `81 + 23×77 = 1,852` | `960×1680` @ `60 fps` (BT.709 High L4.2) | `115.70 s` | **Bundled** (`postprocess_24R_V109.py`) | **Current** |
| **V92 (Baseline)** | `832×1232` @ `30 fps` | `81 + 23×77 = 1,852` | `832×1232` @ `30 fps` | `61.73 s` | External toolchain (not bundled) | Archived baseline |

---

## V109 Architecture & Fixes

V109 converges generation sampling stability, full-body prompt engineering, and 4-stage post-processing fixes into a clean reproducible pipeline:

### 1. P1 Golden Sampling Kernel Restoration
Controlled ablation across R01–R09 established that experimental sweeps (`add_reference_latents=True`, `steps=12`, `MLC=2`, `latent_refresh=0.65`) degraded long-horizon stability:
- **`add_reference_latents = False`**: Enabling reference latents projected ghosted silhouettes/halos from the base image into upper-frame motion and introduced background sky grid artifacts.
- **`steps = 10 (2+4+4)`**: Pushing LightX2V 4-step LoRAs beyond 10 total steps (`steps=12`) over-fitted denoising and amplified checkerboard texture.
- **`motion_latent_count (MLC) = 1`, `lock_start_slots = 1`**: Matches `DROP_HEAD_FROM_R2 = 4` seam slicing (`1,852` total frames across 24 rounds).
- **`latent_refresh = 0.80`, `motion = 1.18`, `CFG = 3.0`, `ImageColorMatch+ (LAB, 0.90)`**: Suppresses dark-area crush and round-to-round color drift while keeping generation outputs separated from the raw `KS3 -> SaveLatent` handoff path.

### 2. Full-Body 9:16 Framing & Motion Guardrails
- Switched `FindPerfectResolution` target to `544×960` at `16 fps` (`81 frames/round ≈ 5.06 s`), expanding a 24-round sequence from ~62 s to **115.70 s**.
- Added full-length framing anchors (`OUTFIT_LOCK` / `DANCE_BASE`) plus anti-crop negative prompts (`cropped feet, feet out of frame, cut off legs, cut off ankles, close-up, medium shot, zooming in`) so lower-body & footwear choreography stays inside the shot across all 24 rounds.
- Removed conflicting slow/soft terms from `MOTION_GUARDRAIL` so it never collides with `FORBIDDEN_ALWAYS` assertions or freezes Round 1 inertia.
- Added strict 24-round validation inside `--dry-run` (all 24 rounds built & asserted before anything is queued).

### 3. 4-Stage Post-Processing Fixes (`postprocess_24R_V109.py`)
- **Stage 1 (`stage1_merged_544x960_16fps.mp4`)**:
  - Per-round single-candidate selection (`round_videos`) prevents duplicate concatenations on retried rounds.
  - `GLOBAL_ANCHOR_WEIGHT = 0.10` + `SEAM_LOCAL_WEIGHT = 0.25` in LAB CDF histogram space, with `seam_blend=False` (`SEAM_MSE_BLEND_MAX = 1500.0`) to eliminate semi-transparent ghosting on moving limbs.
- **Stage 2 (`stage2_upscaled_960x1680_16fps.mp4`)**:
  - Chunked (`chunk_size=64`) 2× upscaling via `RealESRGAN_x2plus.pth` + Lanczos exact resize to `960×1680`, followed by `ImageCASharpening+` (`amount=0.20`), `INTERMEDIATE_CRF=12`, and `video/h264-mp4-high`.
- **Stage 3 (Optional, skipped by default in `--stage all`)**:
  - Per-frame list iteration (`ImpactImageBatchToImageList` → `AILab_YoloV8` `mask_count="2"` → `SEGSDetailer` → `SEGSPaste` → `ImageListToImageBatch`) to avoid batch-index mismatch; skipped by default (`stages = [1, 2, 4]`) to prevent SDXL frame-to-frame micro-flicker.
- **Stage 4 (`stage4_rife_960x1680_64fps.mp4` → `FINAL_960x1680_60fps.mp4`)**:
  - Chunked 4× `FL_RIFE` (`rife49`, `ensemble=True`) with **`RIFE_OVERLAP_TRIM = 1`** (fixing an off-by-3 bug where trimming 4 frames dropped 3 interpolated frames at every 64-frame chunk boundary!).
  - `--master-grade` defaults to `False` (disables legacy fogging/desaturation curves), `--trim-duration` defaults to `-1.0` (preserves full 24-round duration), and final output uses `libx264 -preset slow -crf 14 -profile:v high -level 4.2 -pix_fmt yuv420p` with full BT.709 color tags (`bt709`) and `-movflags +faststart`.

### 4. Local Proxy Isolation & Configurable Environment
- Both Python scripts use `urllib.request.ProxyHandler({})` for ComfyUI HTTP calls so local/system HTTP proxies never intercept loopback (`127.0.0.1`) API traffic.
- Machine-specific paths are fully parameterized via CLI arguments (`--comfy-root`, `--comfy-url`, `--output-dir`, `--root`, `--base-image`) and environment variables (`COMFY_ROOT`, `COMFY_URL`, `FUKE_OUTPUT_DIR`, `ORIGINAL_BASE`, `PY_COMFY`).

---

## Bundled Files

| File | Purpose |
|---|---|
| [`run_fuke_shortsegV109.py`](run_fuke_shortsegV109.py) | V109 24-round ComfyUI generator (`544×960 @ 16fps`, latent chaining, pop-QC retries) |
| [`postprocess_24R_V109.py`](postprocess_24R_V109.py) | V109 4-stage post-processor (`Stage 1` seam match → `Stage 2` 2× upscale+CAS → optional `Stage 3` → `Stage 4` RIFE 4× + BT.709 60fps delivery) |
| [`workflow_fuke_shortsegV109_bootstrap.json`](workflow_fuke_shortsegV109_bootstrap.json) | ComfyUI API workflow for Round 1 (bootstrap from anchor image) |
| [`workflow_fuke_shortsegV109_continue.json`](workflow_fuke_shortsegV109_continue.json) | ComfyUI API workflow for Rounds 2–24 (`LoadLatent` chaining) |
| [`run_fuke_shortsegV92_Claude.py`](run_fuke_shortsegV92_Claude.py) | Legacy V92 (`832×1232 @ 30fps`) generator |
| [`workflow_fuke_shortsegV92_Claude_bootstrap.json`](workflow_fuke_shortsegV92_Claude_bootstrap.json) | Legacy V92 bootstrap API workflow |
| [`workflow_fuke_shortsegV92_Claude_continue.json`](workflow_fuke_shortsegV92_Claude_continue.json) | Legacy V92 continue API workflow |

---

## Requirements

### Hardware
- **GPU**: NVIDIA RTX 3090 (24 GB VRAM) or equivalent — fits the `fp8_scaled` 14B UNETs + rank-128 SVI v2 PRO LoRAs + LightX2V LoRAs and 64-frame post-process chunks.
- **System RAM**: 32 GB+ recommended.
- **Disk**: ~45 GB for Wan 2.2 base models, CLIP/VAE, LoRAs, and post-processing models.

### Software
- **ComfyUI** (with custom nodes listed below).
- **Python 3.10+** with `opencv-python` and `numpy` (used for automatic pop/teleport QC in `run_fuke_shortsegV109.py` and Stage 1 processing in `postprocess_24R_V109.py`).
- **`ffmpeg` & `ffprobe`** available on `PATH`.

---

## Custom Nodes

### Required for Generation & Default Post-Processing (`--stage all`: Stages 1, 2, 4)

| Pack | Repository | Licence | Nodes Used |
|---|---|---|---|
| **IAMCCS-nodes** | https://github.com/IAMCCS/IAMCCS-nodes | GPL | `IAMCCS_WanImageMotion` |
| **ComfyUI-KJNodes** | https://github.com/kijai/ComfyUI-KJNodes | GPL | `PathchSageAttentionKJ`, `ModelPatchTorchSettings` |
| **VideoHelperSuite** | https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite | GPL | `VHS_LoadVideo`, `VHS_VideoCombine` |
| **comfyui_essentials** | https://github.com/cubiq/comfyui_essentials | MIT | `ImageColorMatch+`, `ImageCASharpening+`, `ModelSamplingSD3` |
| **comfyui-find-perfect-resolution** | https://github.com/ashtar1984/comfyui-find-perfect-resolution | *(unspecified)* | `FindPerfectResolution` |
| **rgthree-comfy** | https://github.com/rgthree/rgthree-comfy | MIT | `Seed` (rgthree) |
| **ComfyUI_Fill-Nodes** | https://github.com/filliptm/ComfyUI_Fill-Nodes | — | `FL_RIFE` (Stage 4) |

### Optional Custom Nodes (for `--stage 3` or `--upscale-mode sdxl`)

| Pack | Repository | Nodes Used |
|---|---|---|
| **ComfyUI-Impact-Pack** | https://github.com/ltdrdata/ComfyUI-Impact-Pack | `ImpactImageBatchToImageList`, `SEGSDetailer`, `SEGSPaste`, `MaskToSEGS`, `ToBasicPipe` |
| **ComfyUI-RMBG** | https://github.com/1038lab/ComfyUI-RMBG | `AILab_YoloV8` |
| **ComfyUI_UltimateSDUpscale** | https://github.com/ssitu/ComfyUI_UltimateSDUpscale | `UltimateSDUpscale` |
| **ComfyUI-GGUF** | https://github.com/city96/ComfyUI-GGUF | Optional if swapping in GGUF UNETs |

Core ComfyUI nodes also used: `UNETLoader`, `CLIPLoader`, `VAELoader`, `LoadImage`, `VAEEncode`, `VAEDecode`, `CLIPTextEncode`, `KSamplerAdvanced`, `SaveLatent`, `LoadLatent`, `UpscaleModelLoader`, `ImageUpscaleWithModel`, `ImageScale`, `ImageListToImageBatch`, `GrowMask`, `CheckpointLoaderSimple`.

---

## Models

No weights are bundled. Place them under `<ComfyUI>/models/`:

| Role | Filename | Target Folder |
|---|---|---|
| UNET (high noise) | `wan2.2/wan2.2/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` | `models/diffusion_models/` |
| UNET (low noise) | `wan2.2/wan2.2/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` | `models/diffusion_models/` |
| Text encoder | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (`type="wan"`) | `models/text_encoders/` or `models/clip/` |
| VAE | `wan_2.1_vae.safetensors` | `models/vae/` |
| LoRA (SVI high) | `wan_22/SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors` | `models/loras/` |
| LoRA (SVI low) | `wan_22/SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors` | `models/loras/` |
| LoRA (Lightning high) | `wan_22/Wan_2_2_I2V_A14B_HIGH_lightx2v_4step_lora_v1030_rank_64_bf16.safetensors` | `models/loras/` |
| LoRA (Lightning low) | `wan_22/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors` | `models/loras/` |
| 2× Upscaler (Stage 2) | `RealESRGAN_x2plus.pth` | `models/upscale_models/` |
| RIFE ckpt (Stage 4) | `rife49` (auto-managed by `FL_RIFE`) | ComfyUI RIFE cache |
| Optional SDXL / YOLO | `novaRealityXL_ilV90.safetensors`, `bbox/hand_yolov8n.pt` | `models/checkpoints/`, `models/ultralytics/` |

> If your local `diffusion_models/` or `loras/` subfolders omit `wan2.2\wan2.2\` or `wan_22\`, update those filename strings in the bootstrap and continue workflow JSONs accordingly.

---

## Quick Start (V109)

1. Put your full-body portrait anchor image in `<ComfyUI>/input/`, e.g. `my_anchor.png` (and adapt `OUTFIT_LOCK`, `SCENE_LOCK`, and `DANCE_BASE` in [`run_fuke_shortsegV109.py`](run_fuke_shortsegV109.py) if using a different character/scene).
2. Configure environment variables or pass CLI flags:

```bash
# Optional environment configuration
set COMFY_ROOT=D:\ComfyUI
set COMFY_URL=http://127.0.0.1:8188
set FUKE_OUTPUT_DIR=./output_fuke_shortsegV109_P1
set ORIGINAL_BASE=my_anchor.png
```

3. Validate all 24 rounds without queueing anything to ComfyUI:

```bash
python run_fuke_shortsegV109.py --dry-run
```

4. Start ComfyUI, run a 6-round preview or all 24 rounds:

```bash
# Preview R01..R06 first
python run_fuke_shortsegV109.py --rounds 6 --base-image my_anchor.png --comfy-root "D:\ComfyUI"

# Full 24-round generation (with pop QC & automatic seed retry)
python run_fuke_shortsegV109.py --base-image my_anchor.png --comfy-root "D:\ComfyUI"
```

Useful runner flags: `--comfy-root DIR`, `--comfy-url URL`, `--output-dir DIR`, `--base-image NAME`, `--tag STR`, `--rounds N`, `--start-round K`, `--motion F`, `--cfg F`, `--drop N`, `--prev-dir DIR`, `--prev-prefix STR`, `--no-qc`, `--no-merge`, `--dry-run`.

---

## Post-Processing Pipeline (`postprocess_24R_V109.py`)

Run the 4-stage pipeline on the generated 24-round directory:

```bash
# Zero-side-effect dry run (prints ComfyUI chunk workflows without connecting)
python postprocess_24R_V109.py --dry-run --stage all --max-chunks 1

# Run full commercial standard chain: Stage 1 -> Stage 2 -> Stage 4
python postprocess_24R_V109.py --stage all \
    --root ./output_fuke_shortsegV109_P1 \
    --comfy-root "D:\ComfyUI" \
    --comfy-url http://127.0.0.1:8188
```

Stage breakdown and outputs under `<root>/_post24R/`:
1. **Stage 1 (`--stage 1`)**: `stage1_merged_544x960_16fps.mp4` — local seam color alignment + `drop=4` stitch.
2. **Stage 2 (`--stage 2`)**: `stage2_upscaled_960x1680_16fps.mp4` — chunked `RealESRGAN_x2plus.pth` 2× upscale + `ImageCASharpening+` (`--sharpen 0.20`).
3. **Stage 3 (`--stage 3`, optional)**: `stage3_detailed_960x1680_16fps.mp4` — per-frame YOLO + SEGSDetailer (`--yolo-models bbox\hand_yolov8n.pt`).
4. **Stage 4 (`--stage 4`)**: `stage4_rife_960x1680_64fps.mp4` → **`FINAL_960x1680_60fps.mp4`** — chunked 4× `FL_RIFE` (`rife49`) with `RIFE_OVERLAP_TRIM=1` followed by commercial `libx264 -preset slow -crf 14 -profile:v high -level 4.2 -pix_fmt yuv420p` BT.709 `60 fps` (`+faststart`) encode.

Additional flags: `--work-dir`, `--run-tag`, `--chunk-size` (default `64`), `--upscale-mode` (`model` | `sdxl`), `--seam-blend`, `--master-grade` (default `False`), `--trim-duration` (default `-1.0`), `--fade-out` (default `2.0`).

---

## Workflow Graph & Parameters

```text
LoadImage (anchor)
      │
      ▼
FindPerfectResolution (544×960 portrait in V109; 832×1232 in V92)
      │
      ▼
   VAEEncode ──────────────────────────┐
      │                                │
UNETLoader ×2 (high + low noise fp8)   │
      │                                │
LoraLoaderModelOnly ×3                 │
  SVI v2 PRO HIGH / LOW (rank 128)     │
  LightX2V 4-step HIGH / LOW (rank 64) │
      │                                │
ModelSamplingSD3 (shift = 8)           │
      │                                │
ModelPatchTorchSettings ×3 (fp16 accum)│
      │                                │
PathchSageAttentionKJ ×2               │
      │                                │
IAMCCS_WanImageMotion ─────────────────┤   (bootstrap: use_prev_samples=False)
  length 81, motion 1.18,              │   (continue:  loads prev latent)
  latent_refresh 0.8, MLC=1,           │
  lock_start_slots=1,                  │
  add_reference_latents=False,         │
  anchor_samples ← VAEEncode           │
      │                                │
KSamplerAdvanced ×3 (euler/simple)     │
  steps 0–2  (CFG 3.0)                 │
  steps 2–6  (CFG 1.0, high)           │
  steps 6–10 (CFG 1.0, low)            │
      │                                │
VAEDecode ×2 ──────────────────────────┘
      │
      ├──► ImageColorMatch+ (LAB 0.90) ──► VHS_VideoCombine (h264-mp4-high)
      └──► SaveLatent (raw KS3 latent) ──► feeds next round
```

---

## Measured Results

### V109 Commercial 60 fps Pipeline (`FINAL_960x1680_60fps.mp4`)

All figures below were measured directly from a complete 24-round V109 production run:

- **Generation QC (`qc_log.csv`)**: `26` logged runs across `24` rounds (`2` automatic pop-QC retries triggered and resolved cleanly); final per-round `max_changed_pct` range `1.25% – 18.0%`, `max_ratio` range `1.25 – 4.76`.
- **Pipeline Stage Artifacts (`ffprobe`)**:

| Stage Output | Resolution | FPS | Frames | Duration | Bitrate | Size | Profile / Color |
|---|---|---|---|---|---|---|---|
| `stage1_merged_544x960_16fps.mp4` | `544×960` | `16` | `1,852` | `115.75 s` | `6.07 Mbps` | `87.8 MB` | `High@L3.1` |
| `stage2_upscaled_960x1680_16fps.mp4` | `960×1680` | `16` | `1,852` | `115.75 s` | `24.55 Mbps` | `355.1 MB` | `High@L4.0` / `bt709` |
| `stage4_rife_960x1680_64fps.mp4` | `960×1680` | `64` | `7,405` | `115.70 s` | `24.56 Mbps` | `355.2 MB` | `High@L4.2` / `bt709` |
| **`FINAL_960x1680_60fps.mp4`** | **`960×1680`** | **`60`** | **`6,942`** | **`115.70 s`** | **`17.73 Mbps`** | **`256.4 MB`** | **`High@L4.2` / `bt709`** |

- **Seam & Interpolation Continuity**:
  - **Stage 1 (23 inter-round seams)**: Inter-frame MSE `62.12 – 133.92` (median `100.92`, mean `99.20`), zero hard resets.
  - **Stage 4 (RIFE chunk 0\|1 boundary, `RIFE_OVERLAP_TRIM = 1`)**: Adjacent frame MSE across frames `250..256 @ 64 fps` measured `51.56 – 62.30` with **`f252→f253 = 56.30`** directly on the chunk boundary (zero dropped frames, zero chunk-edge stutter).
- **Full-Duration Tonal Stability, Sky Cleanliness & Sharpness (`t = 1.0 s – 112.5 s`)**:
  - `LAB L` mean: `137.1 – 139.0` (virtually flat tonal stability across all 24 rounds)
  - `HSV Sat` mean: `93.8 – 99.0`
  - `Laplacian Variance` (sharpness): `301.9 – 358.5`
  - `SkyGridStd` (high-pass background sky patch std): `0.74 – 5.01`
  - `Blk<=12` (global shadow share): `2.79% – 5.20%`; `BamCrush<=8` (shadow bamboo foliage crush): `0.00% – 0.03%`

<details>
<summary>V92 Legacy Baseline Measurements (<code>832×1232 @ 30 fps</code>, 61.73 s)</summary>

- Seam repair median speed ratio improved `0.198 → 0.943`, seam colour step `|dL|` reduced `0.294 → 0.143`, `0 / 60` scene-change spikes on seams.
- Round-to-round `LAB dL` spread reduced `0.560 → 0.0936`; global residual `0.0019` vs reference.
- Container: `1,852` frames @ `30 fps`, `61.73 s`, `14.59 Mbps`, H.264 High, `faststart`, zero black/frozen segments.

</details>

---

## Changelog

| Version | Date | Highlights |
|---|---|---|
| **V109** | 2026-11-19 | Restored **P1 golden sampling kernel** (`steps 2+4+4=10`, `MLC=1`, `lock_start_slots=1`, `add_reference_latents=False`, `lr=0.80`, `motion=1.18`, `LAB 0.90`) combined with V108 full-body anti-crop prompts and fixed `MOTION_GUARDRAIL`. Bundled full 4-stage post-processor [`postprocess_24R_V109.py`](postprocess_24R_V109.py) with `RIFE_OVERLAP_TRIM=1`, `ImageCASharpening+ (0.20)`, `INTERMEDIATE_CRF=12`, legacy `--master-grade` disabled by default, full-length preservation (`--trim-duration -1`), BT.709 commercial H.264 `60 fps` output (`960×1680`), env/CLI path parameterization, and loopback proxy bypass. |
| **V108** | 2026-11-19 | Added full-body + footwear anti-crop framing locks/negatives, fixed slow-word guardrail collision, added `ImageCASharpening+`, fixed `RIFE_OVERLAP_TRIM=1`, removed dead `color_lock_tail_frame` path. |
| **V106–V107** | 2026-11-18 | Transitioned to 9:16 `544×960 @ 16 fps` full-body generation (~5.06s/round) and built the chunked 4-stage ComfyUI + ffmpeg post-processing pipeline (`RealESRGAN_x2plus`, optional `SEGSDetailer`, `FL_RIFE`). Identified regressions from experimental `add_reference_latents=True` / `steps=12` / `MLC=2` / `lr=0.65` and reverted in V109. |
| **V92** | 2026-11-14 | Initial open-source release: section-structured 24-round anime choreography (`832×1232 @ 30fps`) with latent bridging, LAB color match, and pop QC. |

---

## License

- **Scripts & Workflow Designs (`*.py`, `workflow_*.json`)**: Released under the **MIT License**.
- **Third-Party Custom Nodes & Model Weights**: Not bundled; observe each upstream repository and model license (GPL / MIT / Apache 2.0 / non-commercial or source terms where applicable).

---

## Credits

- **Upstream Models & Nodes**: Wan 2.2, LightX2V, SVI v2 PRO, IAMCCS-nodes, ComfyUI-KJNodes, VideoHelperSuite, comfyui_essentials, ComfyUI_Fill-Nodes (`FL_RIFE`), ComfyUI-Impact-Pack, ComfyUI-RMBG, ComfyUI_UltimateSDUpscale, and rgthree-comfy.
