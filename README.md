# V92 — Wan 2.2 I2V · 24-Round Multi-Round Video Pipeline

**Seam-repaired, colour-calibrated, and QC-proven.**

24 rounds · 1,852 frames · 61.73 s · 832×1232 @ 30 fps · one continuous dance, no cuts, no loop.

V92 is a full multi-round image-to-video pipeline for **Wan 2.2 I2V-A14B**. It generates 24 rounds
that chain off each other through a shared latent, then runs a post-processing chain that repairs the
seams, calibrates the colour against the reference image, and measures the result. The QC numbers
shipped here are measured **on the delivered file**, not predicted by the script that made it.

> 本仓库是 V92 的完整发布线：Wan 2.2 I2V-A14B 多轮长视频流水线，24 轮、约 62 秒、一镜到底。
> 重点解决三个问题：**接缝可见**（光流诊断 + 重定时）、**轮间色彩漂移**（逐轮 LAB 匹配 + 全局校色）、
> **动作重复**（按段落结构编排，每轮换身体部位与机位）。

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Custom Nodes](#custom-nodes)
- [Models](#models)
- [Workflow](#workflow)
- [Parameters](#parameters)
- [Results](#results)
- [Changelog](#changelog)
- [License](#license)
- [Contact](#contact)

---

## Overview

Multi-round I2V workflows usually break in one of three places: the join between rounds stalls or
jumps, the colour drifts round over round, and every round repeats the same gesture. V92 is built
around those three failure modes specifically, and ships the QC that proves each one was addressed.

| | |
|---|---|
| **Base model** | Wan 2.2 I2V-A14B (fp8 scaled), high + low noise UNET |
| **Rounds** | 24 (round 1 = bootstrap, rounds 2–24 continue from the previous latent) |
| **Frame schedule** | `81 + 23×77 = 1,852` frames (4 head frames dropped from R02+) |
| **Output** | 832×1232, 30 fps, h264 High / yuv420p, 14.59 Mbps, faststart |
| **Duration** | 61.73 s |
| **Audio** | none — the film is silent; add music after |
| **Hardware used** | RTX 3090 (24 GB) |

### The three problems this solves

**1. Visible joins.** Naive round-to-round concatenation freezes at every boundary — the camera
appears to stall, then lurch. V92 diagnoses each join by optical flow and **re-times the seam
window** so the join moves at the same speed as the surrounding footage.

**2. Colour drift between rounds.** Each round drifts a little further in hue and brightness, so the
joins show a colour step. Every round is matched to the reference image through ComfyUI's
`ImageColorMatch+` (LAB, factor 0.9), then the finished film gets one **global LAB offset**
calibrated against the reference. A global offset adds no seam, unlike per-round offsets.

**3. The "same pose every round" problem.** A 24-round video where the dancer repeats one gesture is
dead on arrival. V92's choreography is written as a **musical structure** — intro, verse,
pre-chorus, chorus, interlude, second chorus, outro — with different camera framing, a different
body part per round, and two recurring motifs that return in a different arrangement each time.

### Choreography structure

| Section | Rounds | Purpose |
|---|---|---|
| Intro | R01–R03 | restrained, confident; introduces motif A |
| Verse | R04–R08 | torso / hip / shoulder groove; R06 breath |
| Pre-chorus | R09–R11 | tension rising; R11 sinks before the drop |
| Chorus 1 | R12–R15 | accent peaks R12, R15; motif A scaled up, motif B appears |
| Interlude | R16–R17 | lower centre of gravity, profile, gaze change |
| Chorus 2 | R18–R21 | accent peaks R18, R21; R19 slow build |
| Outro | R22–R24 | R23 final accent (motif A, two-hand), R24 resolves |

- **Motif A** — *"her right hand draws a line from her hip up to her cheek and pushes out toward the
  camera"* — returns at R01 / R12 / R23, each time paired with different movement.
- **Motif B** — *"her arms cross in front of her chest and snap open wide"* — returns at R05 / R15 / R21.
- **Round classes** — `M` mid-tempo ×8, `Q` quicker ×6, `P` accent peak ×5, `S` slower ×5.
  `powerful` appears only in `P` rounds, and `P` rounds are never closer than 2 rounds apart.

---

## Features

- **Latent bridging** — round N+1 loads round N's saved latent (`LoadLatent`) and continues from it,
  so the motion carries across the boundary instead of restarting.
- **Per-round colour match** — `ImageColorMatch+` in LAB at factor 0.9, applied every round.
- **Global colour calibration** — one LAB offset on the finished film, matched to the reference
  image; lands within **0.0019** LAB L of the reference (tolerance 0.05).
- **Optical-flow seam retiming** — the seam window is re-timed so join speed matches interior
  motion instead of stalling or jumping.
- **Head-frame trim sweep** — the drop count is *measured*, not assumed: a sweep over trim 0…12
  showed **trim = 4 is the only value** that brings join error below the film's own interior motion.
- **Automatic QC + retry** — each round is checked for large-change pixel pop and for near-duplicate
  frames; a failing round is re-seeded (`MAX_RETRIES = 2`, `QC_RETRIES = 2`).
- **Structured prompt vocabulary** — forbidden-term lists per round class, so a round cannot drift
  outside its intended motion family.
- **Full QC workbook** — a 15-sheet Excel report where every figure traces back to a raw CSV/JSON
  in `data/`.

---

## Requirements

### Hardware

| | |
|---|---|
| **GPU** | RTX 3090 (24 GB) — the fp8 UNETs plus the rank-128 LoRAs are the memory ceiling |
| **VRAM fallback** | drop the LightX2V-only path if you are tight |
| **RAM** | 32 GB+ recommended (the post-processing chain holds full-frame LAB tensors) |
| **Disk** | ~45 GB for the models listed below |

### Software

| Component | Version / note |
|---|---|
| **ComfyUI** | ak3 (秋葉 aaaki v3) or upstream ComfyUI |
| **Python (generation)** | the ComfyUI venv — needs `kornia`, `mediapipe` |
| **Python (post-processing)** | a second interpreter — needs `opencv-python`, `numpy`, `openpyxl`, `torch` |
| **ffmpeg / ffprobe** | on `PATH` — the seam and QC tooling shells out to both |

The two Python interpreters are separate on purpose: the ComfyUI venv carries `kornia`/`mediapipe`
but not `openpyxl`, and the post-processing venv carries `openpyxl` but not `kornia`. `v92_pipeline.py`
takes both paths as constants (`PY_SYS`, `PY_COMFY`) and dispatches each tool to the interpreter that
actually has its dependencies.

---

## Installation

### 1. Custom nodes

Install the packs listed in [Custom Nodes](#custom-nodes) via ComfyUI Manager, or clone them into
`<ComfyUI>/custom_nodes/`.

### 2. Models

Place the files listed in [Models](#models) into the matching `<ComfyUI>/models/` subfolder.
A `put_vae_here` placeholder file in `models/vae/` marks that folder as the VAE target.

### 3. Reference image

Put your anchor image in `<ComfyUI>/input/` and set its filename in the runner:

```python
ORIGINAL_BASE = "your_anchor_image.jpg"
```

The reference image drives the whole film's identity, costume and palette — it is the image every
round is colour-matched against. The image shipped with this pack was made with **Grok (xAI)**;
swap it for your own, but **if you change the face, re-verify the colour baseline** — the global
calibration offset is computed against that specific image.

### 4. Paths in the runner

`run_fuke_shortsegV92_Claude.py` has three machine-specific constants near the top:

```python
API_URL    = "http://127.0.0.1:8188/prompt"   # ComfyUI must be running
COMFY_ROOT = r"<path to your ComfyUI>"        # parent of the ComfyUI folder
OUTPUT_DIR = r"<path where rounds are written>"
```

`COMFY_ROOT` and `OUTPUT_DIR` are also readable from the `COMFY_ROOT` / `FUKE_OUTPUT_DIR`
environment variables in the sanitized upload copy.

### 5. Generate

Start ComfyUI, then:

```bash
# look at the first 6 rounds before committing to all 24
python run_fuke_shortsegV92_Claude.py --rounds 6

# dry run — validate prompts and workflow wiring, generate nothing
python run_fuke_shortsegV92_Claude.py --dry-run

# full run (24 rounds, with automatic QC + retry)
python run_fuke_shortsegV92_Claude.py
```

Useful flags: `--tag S` · `--motion F` · `--cfg F` · `--base-image NAME` · `--start-round K` ·
`--drop N` · `--no-qc` · `--no-merge`.

> **Framing note.** The shipped anchor is framed head-to-thigh, so V92's choreography only uses body
> parts that are actually visible in frame — arms, shoulders, chest, torso, hips, head, gaze, hands.
> To do full-body dance you need a full-length anchor image. The runner has a `--base-image` switch
> for this, but changing the image changes how much of the face occupies the frame, so the colour and
> identity baseline must be re-verified.

### 6. Post-process

`v92_pipeline.py` drives the whole repair chain. Steps run in this order and can be run individually:

```bash
python v92_pipeline.py --step all              # everything
python v92_pipeline.py --step cm,merge         # just two steps
```

| Step | Does |
|---|---|
| `inventory` | inventory the source rounds, verify the frame schedule against the baseline |
| `probe` | measure per-round colour statistics against the reference |
| `cm` | pick the colour-match factor and apply per-round matching |
| `stage` | stage the rounds into flat segments for measurement |
| `diagnose` | diagnose each join, sweep the head-trim value |
| `merge` | re-time the seams and merge into the master |
| `delivery` | apply the global LAB calibration, encode the delivery file |
| `verify` | QC the merged file: seam metrics, scene spikes, container audit |
| `verify_cm` | re-measure colour after matching |
| `measure` | optical flow, limb amplitude, pose repetition, flow decomposition |
| `report` | build the Excel workbook + Markdown report |

Every step records its status, return code and log path to `_pipeline_state.json`, so an
interrupted run resumes where it stopped.

### 7. Build the report

```bash
python v92_build_report.py --root <repair_dir> --post <post_dir> --out <report.md> --outbox <delivery_dir>
```

Produces `V92Claude_24R_交付报告.xlsx` (15 sheets), a Markdown repair report, and the `data/` folder
holding every raw CSV/JSON the report is built from.

---

## Custom Nodes

These are the packs V92 actually loads. Each node in the workflow is traced to its owning pack:

| Pack | GitHub | Licence | Provides |
|---|---|---|---|
| **IAMCCS-nodes** | https://github.com/IAMCCS/IAMCCS-nodes | GPL | `IAMCCS_WanImageMotion` — the motion/latent-bridge node |
| **ComfyUI-KJNodes** | https://github.com/kijai/ComfyUI-KJNodes | GPL | `PathchSageAttentionKJ`, `ModelPatchTorchSettings` |
| **VideoHelperSuite** | https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite | GPL | `VHS_VideoCombine` |
| **comfyui_essentials** | https://github.com/cubiq/comfyui_essentials | MIT | `ImageColorMatch+`, `ModelSamplingSD3` |
| **comfyui-find-perfect-resolution** | https://github.com/ashtar1984/comfyui-find-perfect-resolution | *(no licence file in the repo)* | `FindPerfectResolution` |
| **rgthree-comfy** | https://github.com/rgthree/rgthree-comfy | MIT | `Seed` (rgthree) |
| **ComfyUI-GGUF** | https://github.com/city96/ComfyUI-GGUF | Apache 2.0 | *optional* — only if you swap in the GGUF UNETs |

Everything else the graph uses is **ComfyUI core**: `UNETLoader`, `CLIPLoader`, `VAELoader`,
`LoadImage`, `VAEEncode`, `VAEDecode`, `KSamplerAdvanced`, `ModelSamplingSD3`, `SaveLatent`,
`LoadLatent`.

> The post-processing chain additionally calls a set of standalone measurement scripts
> (`probe_round_color.py`, `fix_seams_retime.py`, `sweep_trim.py`, `qc_merged.py`,
> `compare_merged_seams.py`, `check_seam_scene_spikes.py`, `round_region_metrics.py`,
> `measure_limb_amplitude.py`, `flow_decomposition.py`, `wrist_local_flow_test.py`,
> `fast_region_displacement.py`, `audit_delivery.py`, `analyze_rounds.py`,
> `reference_color_match.py`, `measure_encode_roundtrip.py`, `run_colormatch.py`,
> `stage_rounds_dir.py`, `seam_jump_analysis.py`). These are the QC toolchain and are **not**
> bundled here — the workflow, the runner and the report builder are; the measurement scripts are
> listed so you know what the numbers in [Results](#results) were produced with.

---

## Models

No weights are bundled. Download them from their official sources and follow their licences.

| Role | File | Put in |
|---|---|---|
| UNET (high noise) | `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` | `models/diffusion_models/` |
| UNET (low noise) | `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` | `models/diffusion_models/` |
| Text encoder | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (type `wan`) | `models/LLM/` |
| VAE | `wan_2.1_vae.safetensors` | `models/vae/` |
| LoRA (high) | `SVI_v2_PRO_Wan2.2-I2V-A14B_HIGH_lora_rank_128_fp16.safetensors` | `models/loras/` |
| LoRA (low) | `SVI_v2_PRO_Wan2.2-I2V-A14B_LOW_lora_rank_128_fp16.safetensors` | `models/loras/` |
| LoRA (lightning, high) | `Wan_2_2_I2V_A14B_HIGH_lightx2v_4step_lora_v1030_rank_64_bf16.safetensors` | `models/loras/` |
| LoRA (lightning, low) | `wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors` | `models/loras/` |

**GGUF alternative** — if you prefer the quantised UNETs, `ComfyUI-GGUF` is enough on its own
(`wan2.2_i2v_high_noise_14B_Q8_0.gguf` / `wan2.2_i2v_low_noise_14B_Q8_0.gguf`), and the `UNETLoader`
node becomes `UnetLoaderGGUF`.

**Anchor image** — one reference image in `<ComfyUI>/input/`. This is the only input the pipeline
takes; it defines the character's identity, costume and palette for all 24 rounds.

---

## Workflow

Two workflow files, in **both API format and UI format**:

| File | Format | Nodes | Links |
|---|---|---|---|
| `workflow_fuke_shortsegV92_Claude_bootstrap.json` | API | 33 | — |
| `workflow_fuke_shortsegV92_Claude_continue.json` | API | 34 | — |
| `workflow_fuke_shortsegV92_Claude_bootstrap_ui.json` | **UI** | 33 | 38 |
| `workflow_fuke_shortsegV92_Claude_continue_ui.json` | **UI** | 34 | 39 |

The `_ui` files are the ones to upload — Civitai renders a UI-format node graph and ignores an
API-format one. Both were round-trip verified (UI → API reproduces the original API JSON field for
field) and link-endpoint validated.

### Graph structure

```
LoadImage (anchor)
      │
      ▼
FindPerfectResolution (832 wide, divisible by 16)
      │
      ▼
   VAEEncode ──────────────────────────┐
      │                                │
UNETLoader ×2 (high + low noise)       │
      │                                │
LoraLoaderModelOnly ×4                 │
  SVI v2 PRO HIGH / LOW (rank 128)     │
  LightX2V 4-step HIGH / LOW (rank 64) │
      │                                │
ModelSamplingSD3 (shift = 8)           │
      │                                │
ModelPatchTorchSettings ×3 (fp16 accum)│
      │                                │
PathchSageAttentionKJ ×2               │
      │                                │
IAMCCS_WanImageMotion ─────────────────┤   (bootstrap: no prev latent)
  length 81, motion 1.18,              │   (continue:  loads prev latent)
  latent_refresh 0.8,                 │
  anchor_samples ← node 17            │
      │                                │
KSamplerAdvanced ×3                    │
  steps 0–2  (CFG 3.0, euler/simple)   │
  steps 2–6  (CFG 1.0)                │
  steps 6–10 (CFG 1.0)                │
      │                                │
VAEDecode ×2 ──────────────────────────┘
      │
      ▼
ImageColorMatch+ (LAB, factor 0.9)  ← against the anchor image
      │
      ├──────────────► SaveLatent  (feeds the next round)
      ▼
VHS_VideoCombine (30 fps, h264-mp4-high)
```

The **bootstrap** workflow encodes the anchor image and generates round 1 from scratch. The
**continue** workflow additionally loads the previous round's saved latent
(`fuke_shortsegV92_Claude_P1_prev.latent`) and enables `use_prev_samples` on the motion node, which
is what makes the rounds chain instead of restarting.

---

## Parameters

### Generation

| Parameter | Value | Where |
|---|---|---|
| Rounds | 24 | `TOTAL_ROUNDS` |
| Frames per round | 81 | `IAMCCS_WanImageMotion.length` |
| Head frames dropped | 4 (from R02+) | `DROP_HEAD_FROM_R2` |
| Output schedule | `81 + 23×77 = 1,852` frames | derived |
| Resolution | 832 × 1232 (aspect 0.675 ≈ 2:3) | `FindPerfectResolution` |
| FPS | 30 | `VHS_VideoCombine.frame_rate` |
| Seed | 123456789 | `Seed (rgthree)`, `SEED_BASE` |
| Sampler / scheduler | `euler` / `simple` | `KSamplerAdvanced` |
| Steps | 3 stages: 0–2, 2–6, 6–10 (10 effective) | `INTConstant` 2 / 6 / 10 |
| CFG | 3.0 (lightning stage), 1.0 (low-noise stages) | `FloatConstant` |
| Shift | 8 | `ModelSamplingSD3` |
| Motion | 1.18 | `MOTION` |
| Latent refresh | 0.8 | `LATENT_REFRESH` |
| Motion latent count | 1 | `IAMCCS_WanImageMotion` |
| Anchor samples | fed from node 17 (`FindPerfectResolution` output) | `IAMCCS_WanImageMotion.anchor_samples` |
| Colour match | LAB, factor 0.9 | `CM_FACTOR` / `ImageColorMatch+` |
| Max retries | 2 (generation), 2 (QC) | `MAX_RETRIES`, `QC_RETRIES` |
| Round timeout | 90 min | `ROUND_TIMEOUT_MIN` |

### Locked prompt blocks

Three prompt blocks stay constant across every round so identity, costume and background cannot
drift:

- **`OUTFIT_LOCK`** — character identity, white hair, red eyes, cyan silk qipao with gold floral
  embroidery, red frog buttons, high mandarin collar, **high side slit**, plus explicit
  no-costume-change / no-hairstyle-change terms.
- **`SCENE_LOCK`** — static camera, locked-off shot, cherry blossom trees perfectly still, only a
  few pink petals drifting down.
- **`NEGATIVE`** — quality, text/watermark, colour, outfit-change, camera-movement, and
  anti-"holding a pose" / anti-jitter terms.

### QC thresholds

| Threshold | Value | Meaning |
|---|---|---|
| `POP_FRAC` | 7.5 % | large-change pixel share floor |
| `POP_RATIO` | 4.0 | share vs. surrounding median floor |
| `STEP_MAX` | 0.25 | residual round-to-round colour step, in 8-bit levels |
| `FLICKER_LOSS_MAX` | 0.25 | mean-L flicker loss budget, in 8-bit levels |
| Seam "seamless" band | 0.80 – 1.25 | join speed ratio vs. surrounding motion |

---

## Results

All figures below are measured **on the delivered file**
(`V92Claude_24R_master_444_refmatched.mp4`, 1,852 frames decoded, 0 decode errors), not predicted.

### Seam repair

| Metric | Before | After |
|---|---|---|
| Join speed ratio vs. surrounding motion (median) | **0.198** | **0.943** |
| Join speed ratio (mean) | 0.217 | 1.048 |
| Join colour step \|dL\| (median) | **0.294** | **0.143** |
| Joins closer to 1.00 after repair | — | **19 / 23** |
| Joins inside the 0.80–1.25 band | 0 / 23 | 9 / 23 |
| Scene-change spikes landing on a join | — | **0 / 60** |

### Colour

| Metric | Before | After |
|---|---|---|
| Round-to-round colour spread (LAB dL) | 0.560 | **0.0936** |
| Round-to-round spread (LAB a) | 0.538 | 0.194 |
| Round-to-round spread (LAB b) | 0.247 | 0.164 |
| Delivered film vs. reference image (LAB L residual) | — | **0.0019** (tolerance 0.05) |

### Motion and amplitude

| Metric | Value |
|---|---|
| Optical flow, mean | 3.28 px/frame |
| Optical flow, person region | 4.04 px/frame |
| Optical flow, background region | 1.22 px/frame |
| Background / person flow ratio | 0.336 |
| Pause ratio (mean / max) | 0.041 / 0.158 |
| Wrist-to-torso reach, max | 1.86 shoulder-widths (mean) |
| Arm span, max | 2.80 shoulder-widths (mean) |
| Elbow extension, max | 176.6° (mean) |
| Pose detection rate | 98.6 % (1 glitch frame, in R23, excluded) |
| Adjacent-round pose distance | 0.529 RMS (non-adjacent 0.536) |

### Delivery file

| | |
|---|---|
| Container | h264 High, yuv420p, faststart, 31 keyframes / 2 s GOP |
| Bitrate | 14.59 Mbps |
| Duration | 61.73 s (1,852 frames @ 30 fps, constant frame rate) |
| Black / frozen segments | 0 / 0 |
| Near-duplicate frame pairs | 0 |

### Honest scope

- **9 of 23** joins sit inside the tight 0.80–1.25 band. **8** remain slightly *slow* (0.34–0.71)
  and **6** slightly *fast* (1.44–2.23). No join is a visible cut any more, but this is not a claim
  of "zero seams".
- The film is **silent** — add music after.
- Aspect ratio is **832×1232 (0.675)**, matching 2:3 rather than 9:16. Crop the width if you need
  exact 9:16.
- 60 frames score above 1.0 on the scene-change scale, and **none of them land on a join** — they
  are choreography beats, not cuts.
- **Content note:** the reference costume is a `high side slit` qipao and the film shows upper-thigh
  skin. The negative prompt does **not** contain slit-suppression terms. Grade the upload
  accordingly.

### Where every number comes from

| Claim | Source |
|---|---|
| Frame schedule, resolution, fps, base image | `data/inventory.json` |
| Seam ratio 0.198 → 0.943, \|dL\| 0.294 → 0.143, 19/23 closer | `data/final_vs_baseline_seams.json` |
| 0 / 60 scene spikes on joins | `data/scene_spike_seams.json` |
| Round-to-round spread 0.560 → 0.0936 | `data/probe_compare.json` |
| Film vs. reference residual 0.0019 | `data/reference_color_match.json` |
| Container / codec / bitrate / black-freeze audit | `data/delivery_audit.json` |
| Trim = 4 is the only seamless value | `data/trim_sweep.json` |
| Limb amplitude, pose detection rate | `data/V92Claude_limb_amplitude_summary.json` |
| Flow by round, flow decomposition | `data/V92Claude_round_region_metrics.csv` / `.json` |

---

## Changelog

| Version | Date | Change |
|---|---|---|
| **V92** | 2026-11-14 | Prompt-only rewrite on top of the G3g generation parameters: 24 rounds arranged as a musical structure, per-round body-part and framing variation, motif A/B recurrence, and the vocabulary check that keeps a round inside its motion family. Generation parameters, workflow graph, colour handling, MLC and QC are unchanged from G3g. |
| **G3i** | 2026-11-13 | Predecessor line — the last G-series runner before the V92 rewrite. |
| **G3h** | 2026-11-12 | Predecessor line. |
| **G3g** | 2026-11-11 | Base of the V92 prompt work; V92 inherits its parameters, workflow, colour handling, MLC and QC verbatim. |
| **G3f** | 2026-11-11 | Added the anti-"holding a pose" and anti-jitter negative-prompt terms. |
| **G3e** | 2026-11-10 | Established the `SCENE_LOCK` / `NEGATIVE` baseline that V92 keeps byte-identical. |
| **V91 (anime)** | 2026-11-11 | Anime-styled variant on the same short-segment pipeline. |

The V92 runner header records the reasoning that drove the rewrite: a review of the earlier cut noted
that colour, identity, costume and background were already good, but the amplitude was too small,
the poses repeated, and there was no sectional variation — and that the composition was head-to-thigh,
so the accent rounds (kicks, steps) were wasting their force outside the frame. V92's choreography
therefore only uses body parts visible in shot.

**Development scale** — the lineage behind this pipeline starts in **June 2026**, when the same core
problem was already being diagnosed: the 81-frame multi-segment structure and the "motion reads as
fast-forward" failure. The runner scripts themselves were written across **September–November
2026**. On the V92 release track alone: **142 runner scripts**, **86 distinct pipeline versions**
(`v2`–`v733`), and **4,787 test artefacts** in the working tree for 2026-09 through 2026-11. Every
one of those was a real generation that had to be watched, measured and thrown away before the next
revision.

---

## License

The work in this repository falls into three categories with three different answers.

**Our own code — MIT License.** The Python scripts in this package
(`run_fuke_shortsegV92_Claude.py`, `v92_pipeline.py`, `v92_build_report.py`,
`v92_factor_final.py`, `v92_factor_clip.py`) and the original workflow design are released under
the MIT License: you may use, modify, redistribute and sublicense them, provided the copyright
notice and this permission notice are included.

```
MIT License

Copyright (c) 2026 V92 contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Third-party custom nodes — keep their licences.** The workflow depends on external node packs
that are **not** bundled here and **not** covered by the MIT grant above:

| Pack | Licence |
|---|---|
| IAMCCS-nodes | GPL |
| ComfyUI-KJNodes | GPL |
| ComfyUI-VideoHelperSuite | GPL |
| comfyui_essentials | MIT |
| rgthree-comfy | MIT |
| ComfyUI-GGUF | Apache 2.0 |
| comfyui-find-perfect-resolution | *(no licence file published — treat as all-rights-reserved)* |

Install them from their own repositories and observe their terms. A workflow that loads GPL nodes is
not thereby relicensed; the GPL obligation attaches to the GPL node code itself.

**Model weights — not packaged, follow the source licence.** No `.safetensors` or `.gguf` file is
included in this repository. The base model, the text encoder, the VAE and all LoRAs are the
property of their respective authors and are redistributed under their own terms. Download them from
their official sources and respect those terms — including any non-commercial or use-restriction
clauses the individual LoRAs carry.

**The generated video.** The shipped film was produced with the models and nodes above. If you
publish it or derivatives of it, credit the underlying models and node authors, and keep the
generated-material terms of the weights you used in mind.

**Trademarks.** "Wan 2.2", "LightX2V", "SVI", "ComfyUI", "Civitai" and the names of the node packs
above belong to their respective owners and are used here only to identify what this pipeline runs on.

---

## Contact

- **Issues with the workflow or the scripts** — open an issue on the repository this package was
  published to.
- **Upstream models and nodes** — each project has its own repository and issue tracker; please
  report model or node problems there rather than here.
- **Business / licensing enquiries** — contact details to be added.

---

## Credits

This pipeline exists because a lot of different hands and models touched it.

- **Claude** — wrote the V92 runner and the whole `G3 → V92` line of prompt choreography: the
  musical section structure, the recurring motifs, the per-round body-part and framing variation, and
  the vocabulary checks that stop a round from drifting outside the intended motion family.
- **GPT** — the creative review that made the rewrite necessary in the first place. Its notes on the
  earlier cut (colour, identity, costume and background were good; amplitude too small, poses
  repetitive, no sectional variation) are quoted at the top of the runner script and are the direct
  cause of the section-based approach in V92.
- **Grok (xAI)** — the reference image.
- **Wan 2.2 / LightX2V / SVI / KJNodes / VideoHelperSuite / IAMCCS / rgthree / comfyui_essentials** —
  the upstream base models, LoRAs and custom nodes this runs on. Please credit and support them.

---

## License summary

| Component | Terms |
|---|---|
| `run_fuke_shortsegV92_Claude.py`, `v92_*.py`, workflow design | MIT (see above) |
| Custom node packs | Their own licences — GPL / MIT / Apache / unspecified |
| Model weights | Not included — their own terms |
| Generated film | Generated with the above; credit the models and node authors |
