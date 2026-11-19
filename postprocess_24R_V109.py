# -*- coding: utf-8 -*-
r"""
postprocess_24R_V109.py
=======================
24R (V109) 四阶段商用级视频后处理管线。**顺序绝对不可颠倒**：

    Stage 1  原分辨率接缝修复与合并      544x960  @16fps   （本地 cv2/numpy + ffmpeg）
    Stage 2  2x 超分辨率放大 + CAS 锐化   544x960 -> 960x1680 @16fps （ComfyUI）
    Stage 3  可选局部修复(脸/手/脚)      960x1680 @16fps          （ComfyUI Detailer）
    Stage 4  4x 光流补帧 + 60fps 商用交付 960x1680 @60fps          （ComfyUI FL_RIFE + ffmpeg BT.709）

为什么是这个顺序（不可调换）：
  * Stage 1 必须在最前：接缝是像素级硬切，后面的超分/修复/补帧都会把它放大成拖影，
    所以必须先在原分辨率把接缝的颜色与时间过渡抹平。
  * Stage 3 必须在 Stage 2 之后：低分辨率下检测器会误判（脸/手太小），超分后的 960x1680
    才能给 Detailer 足够像素做细节重建（注：--stage all 默认跑 1->2->4，跳过 Stage 3）。
  * Stage 4 必须最后：补帧依赖光流连续性，任何画面修改之后再补帧才是正确顺序。

Stage 1 完全本地执行（opencv/numpy/ffmpeg），不依赖 ComfyUI。
Stage 2/3/4 通过 ComfyUI HTTP API 执行（默认 http://127.0.0.1:8188）。

环境变量（可选覆盖）：
  COMFY_ROOT       ComfyUI 安装父目录（默认 C:\ComfyUI）
  COMFY_URL        ComfyUI HTTP 地址（默认 http://127.0.0.1:8188）
  FUKE_OUTPUT_DIR  24 轮生成目录默认路径（默认 ./output_fuke_shortsegV109_P1）

用法
----
    # 只看将要提交的工作流（不连 ComfyUI、不创建目录、不渲染）
    python postprocess_24R_V109.py --dry-run --stage all

    # 分步执行
    python postprocess_24R_V109.py --stage 1 --root ./output_fuke_shortsegV109_P1
    python postprocess_24R_V109.py --stage 2 --root ./output_fuke_shortsegV109_P1
    python postprocess_24R_V109.py --stage 4 --root ./output_fuke_shortsegV109_P1

    # 一键执行商用标准链 1->2->4
    python postprocess_24R_V109.py --stage all \
        --root "./output_fuke_shortsegV109_P1" \
        --comfy-root "C:\ComfyUI" \
        --comfy-url http://127.0.0.1:8188 \
        --upscale-mode model

边界说明
--------
  * 本脚本不做任何云端 API 调用；Stage 2/3/4 只与指定 ComfyUI 实例通信。
  * 仅在显式指定对应 --stage 时才执行对应阶段。
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# 常量（与 run_fuke_shortsegV109.py 保持一致）
# --------------------------------------------------------------------------- #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.environ.get("FUKE_OUTPUT_DIR", os.path.join(SCRIPT_DIR, "output_fuke_shortsegV109_P1"))
COMFY_ROOT = os.environ.get("COMFY_ROOT", r"C:\ComfyUI")
COMFYUI_INPUT = os.path.join(COMFY_ROOT, "ComfyUI", "input")
COMFYUI_OUTPUT = os.path.join(COMFY_ROOT, "ComfyUI", "output")
COMFY_URL_DEFAULT = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")

SRC_W, SRC_H = 544, 960
DST_W, DST_H = 960, 1680
FPS = 16
RIFE_MULT = 4
FINAL_FPS = 60

DROP_HEAD_FROM_R2 = 4          # 匹配 Wan2.2/2.1 VAE MLC=1 / lock_start_slots=1 (R02~R24 各砍头 4 帧)
ROUNDS = 24
WINDOW = 2                     # 接缝前后各 2 帧做软过渡
SEAM_HIST_STRENGTH = 1.0       # 0=关闭直方图对齐, 1=完全对齐

# ---- 单轮单视频 + 轻量全局色彩锚点 + 防重影接缝 ----
ROUND_FRAMES = 81              # 每轮有效视频帧数（R01 固定 81 帧）
GLOBAL_ANCHOR_WEIGHT = 0.10    # 向全局基准（R01）做直方图对齐的权重（生成端已有 LAB 0.90）
SEAM_LOCAL_WEIGHT = 0.25       # 接缝处向 prev_tail 局部微调的权重（纯色彩，不做像素叠加）
SEAM_MSE_BLEND_MAX = 1500.0    # 接缝 MSE 超过此值视为大位移，跳过像素淡入叠加（防重影）

STAGE2_SHARPEN_AMOUNT = 0.20   # Stage 2 放大后轻量 CAS 锐化补偿（0=关闭）
RIFE_OVERLAP_TRIM = 1          # FL_RIFE 输出 (N-1)*mult+1 帧，重叠仅需裁掉重复首帧 1 帧
INTERMEDIATE_CRF = 12          # 中间分片与 VHS_VideoCombine 统一高画质 CRF

CKPT = "novaRealityXL_ilV90.safetensors"     # 可选 Stage 3 / sdxl 模式的写实底模
UPSCALE_MODEL = "RealESRGAN_x2plus.pth"      # Stage 2 默认 2x 超分模型
RIFE_CKPT = "rife49"

PREFIX = "FUKE_ShortSegV109_P1"


# --------------------------------------------------------------------------- #
# 通用小工具
# --------------------------------------------------------------------------- #
def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def run(cmd, **kw):
    log("exec: %s" % " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def which_or_die(name):
    p = shutil.which(name)
    if not p:
        raise RuntimeError("required tool not in PATH: %s" % name)
    return p


def ffprobe_size(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,r_frame_rate",
                        "-of", "csv=p=0", path],
                       capture_output=True, text=True, check=True)
    parts = [x.strip() for x in r.stdout.strip().split(",")]
    w, h = int(parts[0]), int(parts[1])
    rate = parts[2] if len(parts) > 2 else "16/1"
    num, _, den = rate.partition("/")
    fps = float(num) / float(den or 1)
    return w, h, fps


def ffprobe_frames(path):
    """获取输入视频的【准确】总帧数。

    优先用 ffprobe -count_frames 精确计数（不依赖容器元数据），
    失败时回退 cv2.VideoCapture(...).get(cv2.CAP_PROP_FRAME_COUNT)。
    返回 0 表示无法确定。
    """
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-count_frames", "-show_entries", "stream=nb_read_frames",
                            "-of", "csv=p=0", path],
                           capture_output=True, text=True, check=True)
        s = r.stdout.strip().splitlines()
        if s and s[0].strip().isdigit():
            n = int(s[0].strip())
            if n > 0:
                return n
    except Exception as e:
        log("ffprobe_frames: ffprobe 计数失败 (%s)，回退 cv2" % e)
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return 0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return n
    except Exception as e:
        log("ffprobe_frames: cv2 回退失败 (%s)" % e)
        return 0


# --------------------------------------------------------------------------- #
# Stage 1 —— 原分辨率接缝修复与合并（纯本地）
# --------------------------------------------------------------------------- #
def _video_frame_count(path):
    """用 CAP_PROP_FRAME_COUNT 读帧数（失败返回 0）。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return n


def round_videos(root, r):
    """返回该轮唯一有效的 .mp4（list，最多 1 个元素）。

    某些 RoundXX 目录下存在多个 .mp4（例如重渲染留下的旧文件），旧逻辑会把
    所有文件顺序拼接，导致前半段与后半段重复从初始底图开始（画面重置跳切）。
    这里只保留【一个】有效视频：
      1) 优先选帧数 == ROUND_FRAMES(81) 的文件；
      2) 否则选修改时间最新的文件；
      3) 否则退化为按文件名排序的最后一个。
    """
    d = os.path.join(root, "Round%02d" % r)
    if not os.path.isdir(d):
        return []
    files = sorted(glob.glob(os.path.join(d, "*.mp4")))
    if not files:
        return []
    if len(files) == 1:
        return files
    exact = [f for f in files if _video_frame_count(f) == ROUND_FRAMES]
    if len(exact) == 1:
        chosen = exact[0]
    elif exact:
        chosen = max(exact, key=os.path.getmtime)
    else:
        chosen = max(files, key=os.path.getmtime)
    log("  Round%02d: %d mp4 candidates -> using %s"
        % (r, len(files), os.path.basename(chosen)))
    return [chosen]


def read_round_frames(root, r):
    """读入某轮全部帧（按文件顺序拼接）。"""
    frames = []
    for v in round_videos(root, r):
        cap = cv2.VideoCapture(v)
        if not cap.isOpened():
            raise RuntimeError("cannot open %s" % v)
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
    return frames


def hist_match_to(src, ref):
    """在 LAB 空间做逐通道 CDF 对齐，避免人物转身改变红绿占比时导致肤色跳变。"""
    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)
    out_lab = np.empty_like(src_lab)
    for c in range(3):
        hist_src = cv2.calcHist([src_lab], [c], None, [256], [0, 256]).ravel()
        hist_ref = cv2.calcHist([ref_lab], [c], None, [256], [0, 256]).ravel()
        cdf_src = hist_src.cumsum() / max(hist_src.sum(), 1.0)
        cdf_ref = hist_ref.cumsum() / max(hist_ref.sum(), 1.0)
        lut = np.interp(cdf_src, cdf_ref, np.arange(256)).astype(np.uint8)
        out_lab[:, :, c] = cv2.LUT(src_lab[:, :, c], lut)
    return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)


def soft_transition(prev_tail, next_head, strength=1.0):
    """接缝软过渡：把 next_head 往 prev_tail 的色调拉，并做线性淡入。

    返回 (新 next_head, 权重列表)。权重用于把淡入渐变施加到接缝后的前 WINDOW 帧。
    """
    if strength <= 0:
        return next_head, [0.0]
    matched = hist_match_to(next_head, prev_tail)
    blended = cv2.addWeighted(matched, strength, next_head, 1.0 - strength, 0)
    return blended, [1.0]


def stage1(root, out_path, drop=DROP_HEAD_FROM_R2, window=WINDOW,
           hist_strength=SEAM_HIST_STRENGTH, seam_blend=False,
           anchor_weight=GLOBAL_ANCHOR_WEIGHT, seam_local_weight=SEAM_LOCAL_WEIGHT,
           seam_mse_max=SEAM_MSE_BLEND_MAX):
    """合并 R01..R24（R01 全帧, R02+ 砍头 drop 帧）。

    V105 修复要点：
      * 每轮只读 1 个有效 mp4（见 round_videos），R01 固定 81 帧，总帧数 81+23*77=1852；
      * R02~R24 的【每一帧】先以 anchor_weight 权重向全局基准 anchor_ref（R01 首帧）
        做逐通道 CDF 直方图对齐，压住暗部死黑 / 高光过曝 / 唇色漂移；
      * 接缝处再以 seam_local_weight 向 prev_tail 做【纯色彩】局部微调；
      * 接缝 MSE 超过 seam_mse_max（大位移/快速动作）时跳过像素淡入叠加，避免半透明重影。
      * V106: 接缝像素淡入叠加默认关闭（seam_blend=False），只保留直方图/LAB 色彩对齐，
        跨帧过渡交由 Stage 4 的 RIFE 补帧平滑完成，消除手部半透明双影。
    """
    which_or_die("ffmpeg")
    log("Stage 1: 读取 24 轮原始帧 ...")
    per_round = []
    for r in range(1, ROUNDS + 1):
        frs = read_round_frames(root, r)
        if not frs:
            raise RuntimeError("no frames for R%02d under %s" % (r, root))
        if r >= 2:
            frs = frs[drop:]
        per_round.append(frs)
        log("  R%02d: %d frames (kept after trim)" % (r, len(frs)))

    # ---- 全局色彩基准 anchor_ref：R01 首帧（保留原始像素，不做任何修改）----
    anchor_ref = per_round[0][0].copy()
    log("Stage 1: 全局色彩基准 anchor_ref = R01 首帧 (anchor_weight=%.2f)" % anchor_weight)

    # ---- 拼接 + 接缝处理 ----
    merged = []
    seam_report = []
    color_report = []
    round_stats = []
    for i, frs in enumerate(per_round):
        if i == 0:
            merged.extend(frs)
            color_report.append((1, 0.0, 0.0))
            continue
        prev_tail = merged[-1]
        head_raw = frs[0]
        # 修复前接缝 MSE：用未修改的原始 frs[0] 与 prev_tail 比较
        seam_mse = float(np.mean((prev_tail.astype(np.float64) - head_raw.astype(np.float64)) ** 2))
        seam_report.append((i + 1, seam_mse))

        # (a) 全局基准对齐：整轮每一帧都向 anchor_ref 做直方图对齐
        if hist_strength > 0 and anchor_weight > 0:
            for k in range(len(frs)):
                matched = hist_match_to(frs[k], anchor_ref)
                frs[k] = cv2.addWeighted(matched, anchor_weight, frs[k], 1.0 - anchor_weight, 0)
        # 记录对齐前后的亮度变化（诊断用）
        b_before = float(np.mean(head_raw))
        b_after = float(np.mean(frs[0]))
        round_stats.append({
            "round": i + 1,
            "frames": len(frs),
            "mean_before": b_before,
            "mean_after": b_after,
            "anchor_ref_mean": float(np.mean(anchor_ref)),
            "black_ratio_before": float(np.mean(np.all(head_raw <= 8, axis=2))),
            "black_ratio_after": float(np.mean(np.all(frs[0] <= 8, axis=2))),
            "seam_mse": seam_mse,
        })

        # (b) 接缝局部微调：向 prev_tail 做纯色彩对齐（不叠加像素，不产生重影）
        if hist_strength > 0 and seam_local_weight > 0:
            for k in range(min(window + 1, len(frs))):
                matched = hist_match_to(frs[k], prev_tail)
                frs[k] = cv2.addWeighted(matched, seam_local_weight, frs[k], 1.0 - seam_local_weight, 0)

        # (c) 像素淡入软过渡：仅在低位移接缝处启用；大位移时跳过，避免双影
        blend_used = False
        if seam_blend:
            if seam_mse > seam_mse_max:
                log("  seam R%02d: MSE=%.1f > %.1f -> 跳过像素淡入（防重影）"
                    % (i + 1, seam_mse, seam_mse_max))
            else:
                w = window
                for k in range(min(w, len(frs))):
                    alpha = 1.0 - (k + 1) / float(w + 1)   # 1st frame 权重最大, 递减到 0
                    alpha = max(alpha, 0.0)
                    frs[k] = cv2.addWeighted(frs[k], 1 - alpha, prev_tail, alpha, 0)
                blend_used = True
        color_report.append((i + 1, b_before, b_after, blend_used, seam_mse))
        merged.extend(frs)

    log("Stage 1: 接缝 MSE（修复前）:")
    for r, v in seam_report:
        log("  seam R%02d|R%02d  MSE = %10.2f" % (r - 1, r, v))

    # 色彩对齐诊断日志（R01 / R12 / R18 / R24 + 全轮）
    log("Stage 1: 色彩对齐日志（anchor_ref=R01 首帧, weight=%.2f）:" % anchor_weight)
    for st in round_stats:
        if st["round"] in (1, 12, 18, 24) or st["round"] == len(per_round):
            log("  R%02d frames=%d mean %.2f -> %.2f (anchor %.2f) | 死黑占比 %.4f -> %.4f | seamMSE %.1f"
                % (st["round"], st["frames"], st["mean_before"], st["mean_after"],
                   st["anchor_ref_mean"], st["black_ratio_before"], st["black_ratio_after"],
                   st["seam_mse"]))
    all_before = float(np.mean([s["mean_before"] for s in round_stats]))
    all_after = float(np.mean([s["mean_after"] for s in round_stats]))
    log("  ALL: mean %.2f -> %.2f | 末轮死黑占比 %.4f" %
        (all_before, all_after, round_stats[-1]["black_ratio_after"]))

    # 写出 544x960@16fps
    h, w = merged[0].shape[:2]
    if (w, h) != (SRC_W, SRC_H):
        merged = [cv2.resize(f, (SRC_W, SRC_H), interpolation=cv2.INTER_AREA) for f in merged]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", "%dx%d" % (SRC_W, SRC_H), "-r", str(FPS), "-i", "-", "-an",
         "-c:v", "libx264", "-crf", str(INTERMEDIATE_CRF), "-pix_fmt", "yuv420p", out_path],
        stdin=subprocess.PIPE)
    for f in merged:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()
    log("Stage 1 DONE -> %s (%d frames, %.2fs @ %dfps)"
        % (out_path, len(merged), len(merged) / float(FPS), FPS))

    # 修复后接缝 MSE 复测
    caps = []
    for i in range(1, len(per_round)):
        idx = sum(len(x) for x in per_round[:i])
        caps.append(idx)
    post = []
    cap = cv2.VideoCapture(out_path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    for i, idx in enumerate(caps):
        a = frames[idx - 1].astype(np.float64)
        b = frames[idx].astype(np.float64)
        post.append((i + 2, float(np.mean((a - b) ** 2))))
    log("Stage 1: 接缝 MSE（修复后）:")
    for r, v in post:
        log("  seam R%02d|R%02d  MSE = %10.2f" % (r - 1, r, v))
    return out_path


# --------------------------------------------------------------------------- #
# ComfyUI HTTP API 封装
# --------------------------------------------------------------------------- #
class Comfy:
    def __init__(self, base="http://127.0.0.1:8188", timeout=600):
        self.base = base.rstrip("/")
        self.timeout = timeout
        # 显式绕过本地/系统 HTTP 代理，确保回环 ComfyUI API 直连
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _get(self, path):
        with self._opener.open(self.base + path, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))

    def _post(self, path, payload):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.base + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with self._opener.open(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def alive(self):
        try:
            self._get("/system_stats")
            return True
        except Exception:
            return False

    def submit(self, workflow):
        r = self._post("/prompt", {"prompt": workflow})
        return r["prompt_id"]

    def wait(self, prompt_id, poll=3.0):
        while True:
            try:
                h = self._get("/history/" + prompt_id)
            except Exception:
                h = {}
            if prompt_id in h and h[prompt_id].get("status", {}).get("completed"):
                return h[prompt_id]
            if prompt_id in h and h[prompt_id].get("status", {}).get("status_str") == "error":
                raise RuntimeError("ComfyUI prompt failed: %s"
                                   % json.dumps(h[prompt_id].get("status"), ensure_ascii=False)[:500])
            time.sleep(poll)

    def queue(self):
        return self._get("/queue")


def comfy_output_latest(prefix, outdir=COMFYUI_OUTPUT, ext=".mp4"):
    """找出 ComfyUI 输出目录里最新生成、文件名以 prefix 开头的文件。"""
    cands = []
    for root, _dirs, files in os.walk(outdir):
        for f in files:
            if f.startswith(prefix) and f.lower().endswith(ext):
                cands.append(os.path.join(root, f))
    if not cands:
        raise RuntimeError("no output %s*%s under %s" % (prefix, ext, outdir))
    return max(cands, key=os.path.getmtime)


def copy_to_comfy_input(src, name=None):
    os.makedirs(COMFYUI_INPUT, exist_ok=True)
    name = name or os.path.basename(src)
    dst = os.path.join(COMFYUI_INPUT, name)
    shutil.copy2(src, dst)
    return name


# --------------------------------------------------------------------------- #
# Stage 2 —— 2x 超分（ComfyUI）
# --------------------------------------------------------------------------- #
def wf_load_video(video_name, frame_rate=FPS, skip_first_frames=0, frame_load_cap=0):
    """VHS_LoadVideo 节点（支持分片：跳过前 N 帧 + 最多加载 M 帧）。"""
    return {"class_type": "VHS_LoadVideo",
            "_meta": {"title": "Load video (544x960@16fps)"},
            "inputs": {"video": video_name, "force_rate": 0,
                       "force_size": "Disabled", "custom_width": 0,
                       "custom_height": 0, "frame_load_cap": int(frame_load_cap),
                       "skip_first_frames": int(skip_first_frames),
                       "select_every_nth": 1}}


def wf_video_combine(node_id, images, prefix, frame_rate=FPS, crf=INTERMEDIATE_CRF):
    return {node_id: {"class_type": "VHS_VideoCombine",
                      "_meta": {"title": "VideoCombine %dfps" % frame_rate},
                      "inputs": {"frame_rate": frame_rate, "loop_count": 0,
                                 "filename_prefix": prefix,
                                 "format": "video/h264-mp4-high",
                                 "pix_fmt": "yuv420p", "crf": int(crf),
                                 "save_metadata": True, "trim_to_audio": False,
                                 "pingpong": False, "save_output": True,
                                 "images": images}}}


# --------------------------------------------------------------------------- #
# 分片（chunk）执行框架 —— 解决一次性加载 1852 帧导致的 ComfyUI 内存溢出
# --------------------------------------------------------------------------- #
CHUNK_SIZE_DEFAULT = 64


def chunk_plan(total, chunk_size):
    """生成 [(offset, count), ...] 分片计划。"""
    plan = []
    off = 0
    while off < total:
        cnt = min(chunk_size, total - off)
        plan.append((off, cnt))
        off += cnt
    return plan


def concat_chunks(chunk_files, out_path):
    """用本地 ffmpeg concat demuxer 无损合成（-c copy）。"""
    which_or_die("ffmpeg")
    list_path = os.path.join(os.path.dirname(os.path.abspath(out_path)),
                             "concat_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for c in chunk_files:
            p = os.path.abspath(c).replace("\\", "/")
            f.write("file '%s'\n" % p)
    try:
        run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", list_path, "-c", "copy", out_path])
    except subprocess.CalledProcessError:
        log("concat -c copy 失败，回退为重编码 concat")
        run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", list_path, "-an", "-c:v", "libx264", "-crf", str(INTERMEDIATE_CRF),
             "-pix_fmt", "yuv420p", out_path])
    log("concat: %d chunks -> %s" % (len(chunk_files), out_path))
    return out_path


def normalize_chunk(src, dst, fps):
    """把分片重编码为统一参数（保证 concat -c copy 可用），可选去掉开头 n 帧。"""
    which_or_die("ffmpeg")
    vf = ("trim=start_frame=%d,setpts=PTS-STARTPTS" % RIFE_OVERLAP_TRIM) if fps > FPS else "null"
    run(["ffmpeg", "-y", "-v", "error", "-i", src, "-vf", vf,
         "-an", "-r", str(fps), "-c:v", "libx264", "-crf", str(INTERMEDIATE_CRF),
         "-pix_fmt", "yuv420p", dst])
    return dst


def trim_leading_frames(src, dst, n, fps):
    """去掉分片开头的 n 帧（重编码，保留 fps），用于 Stage 4 交界去重。"""
    which_or_die("ffmpeg")
    run(["ffmpeg", "-y", "-v", "error", "-i", src,
         "-vf", "trim=start_frame=%d,setpts=PTS-STARTPTS" % n,
         "-an", "-r", str(fps), "-c:v", "libx264", "-crf", str(INTERMEDIATE_CRF),
         "-pix_fmt", "yuv420p", dst])
    return dst


def _ensure_overlap_trimmed(cpath, tpath, trim_n, fps):
    """确保 Stage 4 非首分片只裁掉重复的起始源帧（trim_n=1），避免残留旧版 4 帧裁切。"""
    raw_n = _video_frame_count(cpath)
    expected_n = max(raw_n - trim_n, 1) if raw_n > 0 else 0
    cur_n = _video_frame_count(tpath) if (os.path.isfile(tpath) and os.path.getsize(tpath) > 0) else 0
    if cur_n == 0 or (expected_n > 0 and cur_n != expected_n):
        trim_leading_frames(cpath, tpath, trim_n, fps)
    return tpath


def run_chunked_stage(comfy, tag, src_video_name, wf_builder, chunk_dir,
                      total_frames, chunk_size, out_path, dry_run=False,
                      overlap=False, fps=FPS, max_chunks=0, run_prefix=PREFIX):
    """按 chunk_size 分片提交 ComfyUI，再本地 concat 合成 out_path。

    overlap=True（Stage 4）：FL_RIFE 对 N 帧输入输出 (N-1)*mult+1 帧；
    第 2 片起 skip_first_frames=offset-1、frame_load_cap=cnt+1，借上一片末帧作为首帧
    生成中间过渡帧；合成前仅去掉与上一片末帧完全重合的首 1 帧（RIFE_OVERLAP_TRIM=1），
    保留新生成的 3 个过渡帧。
    """
    if not dry_run:
        os.makedirs(chunk_dir, exist_ok=True)
    plan = chunk_plan(total_frames, chunk_size)
    if max_chunks and max_chunks > 0:
        plan = plan[:max_chunks]
    log("%s: %d frames -> %d chunks (chunk_size=%d, overlap=%s)"
        % (tag, total_frames, len(plan), chunk_size, overlap))

    chunk_files = []
    for idx, (offset, cnt) in enumerate(plan):
        cpath = os.path.join(chunk_dir, "chunk_%03d.mp4" % idx)
        if not dry_run and os.path.isfile(cpath) and os.path.getsize(cpath) > 0:
            log("%s: chunk_%03d exists (%.1f MB) -> skip (断点续跑)"
                % (tag, idx, os.path.getsize(cpath) / 1048576.0))
            use = cpath
            if overlap and offset > 0:
                tpath = os.path.join(chunk_dir, "chunk_%03d_trim.mp4" % idx)
                use = _ensure_overlap_trimmed(cpath, tpath, RIFE_OVERLAP_TRIM, fps)
            chunk_files.append(use)
            continue

        if overlap and offset > 0:
            skip = offset - 1
            cap = cnt + 1
        else:
            skip = offset
            cap = cnt

        prefix = "%s_%s_c%03d_" % (run_prefix, tag, idx)
        wf = wf_builder(src_video_name, prefix, skip, cap)
        out = run_comfy_stage(comfy, wf, prefix, dry_run=dry_run)
        if dry_run:
            continue
        shutil.copy2(out, cpath)
        use = cpath
        if overlap and offset > 0:
            # FL_RIFE 尾帧后无过渡帧；仅去掉与上一分片末帧重合的 idx=0（1 帧），保留后续过渡帧
            tpath = os.path.join(chunk_dir, "chunk_%03d_trim.mp4" % idx)
            use = _ensure_overlap_trimmed(cpath, tpath, RIFE_OVERLAP_TRIM, fps)
        chunk_files.append(use)
        log("%s: chunk_%03d done (offset=%d, skip=%d, cap=%d) -> %s"
            % (tag, idx, offset, skip, cap, use))

    if dry_run:
        return None
    if not chunk_files:
        raise RuntimeError("%s: no chunk produced" % tag)
    concat_chunks(chunk_files, out_path)
    return out_path


def build_wf_stage2(video_name, mode="model", prefix=PREFIX + "_S2_960x1680_16fps",
                    skip_first_frames=0, frame_load_cap=0,
                    sharpen_amount=STAGE2_SHARPEN_AMOUNT):
    """Stage 2: 544x960 -> 960x1680 @16fps 的 2x 超分 + 轻度 CAS 锐化补偿。

    mode="model" : UpscaleModelLoader(UPSCALE_MODEL) + ImageUpscaleWithModel -> ImageScale 到 960x1680
                   （快、稳、尺寸严格可控）
    mode="sdxl"  : UltimateSDUpscale(upscale_by=2.0) 扩散超分（慢、细节更足）
    """
    wf = {}
    wf["1"] = wf_load_video(video_name, skip_first_frames=skip_first_frames,
                            frame_load_cap=frame_load_cap)
    if mode == "sdxl":
        wf["10"] = {"class_type": "CheckpointLoaderSimple",
                    "_meta": {"title": "SDXL realistic checkpoint"},
                    "inputs": {"ckpt_name": CKPT}}
        wf["11"] = {"class_type": "CLIPTextEncode",
                    "_meta": {"title": "Positive"},
                    "inputs": {"text": "masterpiece, best quality, 8k, ultra detailed, "
                                       "realistic skin texture, natural human eyes, "
                                       "sharp focus, high detail",
                               "clip": ["10", 1]}}
        wf["12"] = {"class_type": "CLIPTextEncode",
                    "_meta": {"title": "Negative"},
                    "inputs": {"text": "worst quality, low quality, blurry, illustration, "
                                       "anime, cartoon, 3d render, cgi, doll, plastic skin",
                               "clip": ["10", 1]}}
        wf["13"] = {"class_type": "UpscaleModelLoader",
                    "_meta": {"title": "RealESRGAN_x2plus"},
                    "inputs": {"model_name": UPSCALE_MODEL}}
        wf["14"] = {"class_type": "UltimateSDUpscale",
                    "_meta": {"title": "Stage 2 UPSCALE 2x -> 960x1680"},
                    "inputs": {"image": ["1", 0], "model": ["10", 0],
                               "positive": ["11", 0], "negative": ["12", 0],
                               "vae": ["10", 2], "upscale_by": 2.0, "seed": 12345,
                               "steps": 20, "cfg": 5.5, "sampler_name": "euler",
                               "scheduler": "simple", "denoise": 0.30,
                               "upscale_model": ["13", 0], "mode_type": "Linear",
                               "tile_width": 512, "tile_height": 512,
                               "mask_blur": 0, "tile_padding": 16,
                               "seam_fix_mode": "Band Pass", "seam_fix_denoise": 0.25,
                               "seam_fix_width": 64, "seam_fix_mask_blur": 32,
                               "seam_fix_padding": 16,
                               "force_uniform_tiles": False, "tiled_decode": True}}
        wf["15"] = {"class_type": "ImageScale",
                    "_meta": {"title": "Guarantee 960x1680"},
                    "inputs": {"image": ["14", 0], "upscale_method": "lanczos",
                               "width": DST_W, "height": DST_H, "crop": "disabled"}}
        out = ["15", 0]
    else:
        wf["10"] = {"class_type": "UpscaleModelLoader",
                    "_meta": {"title": "RealESRGAN_x2plus"},
                    "inputs": {"model_name": UPSCALE_MODEL}}
        wf["11"] = {"class_type": "ImageUpscaleWithModel",
                    "_meta": {"title": "Model upscale (2x)"},
                    "inputs": {"upscale_model": ["10", 0], "image": ["1", 0]}}
        wf["12"] = {"class_type": "ImageScale",
                    "_meta": {"title": "Scale to 960x1680"},
                    "inputs": {"image": ["11", 0], "upscale_method": "lanczos",
                               "width": DST_W, "height": DST_H, "crop": "disabled"}}
        out = ["12", 0]
    if sharpen_amount and float(sharpen_amount) > 0:
        wf["16"] = {"class_type": "ImageCASharpening+",
                    "_meta": {"title": "CAS Sharpen (%.2f)" % float(sharpen_amount)},
                    "inputs": {"image": out, "amount": float(sharpen_amount)}}
        out = ["16", 0]
    wf.update(wf_video_combine("20", out, prefix, FPS))
    return wf


# --------------------------------------------------------------------------- #
# Stage 3 —— 高分辨率局部修复（ComfyUI Detailer）
# --------------------------------------------------------------------------- #
def build_wf_stage3(video_name, yolo_models,
                    prefix=PREFIX + "_S3_960x1680_16fps",
                    skip_first_frames=0, frame_load_cap=0):
    """Stage 3: 960x1680 分辨率下的脸/手/脚 Detailer 修复。

    检测器看到的是**已超分**的 960x1680 画面，因此必须在 Stage 2 之后。

    V107.1 批次拆分 -> 列表 -> 逐帧迭代 -> 重新合批：
      1" VHS_LoadVideo 输出整批帧 ->
      6" ImpactImageBatchToImageList 拆成逐帧列表 ->
      检测/SEGSDetailer/SEGSPaste 链按帧迭代（这些节点没有 INPUT_IS_LIST）->
      299" ImageListToImageBatch 重新合批 ->
      300" VHS_VideoCombine 输出视频。
    """
    wf = {}
    wf["1"] = wf_load_video(video_name, skip_first_frames=skip_first_frames,
                            frame_load_cap=frame_load_cap)
    wf["2"] = {"class_type": "CheckpointLoaderSimple",
               "_meta": {"title": "SDXL realistic checkpoint"},
               "inputs": {"ckpt_name": CKPT}}
    wf["3"] = {"class_type": "CLIPTextEncode",
               "_meta": {"title": "Positive"},
               "inputs": {"text": "masterpiece, best quality, 8k, ultra detailed, "
                                  "realistic skin texture, natural pores, sharp focus, "
                                  "correct human anatomy, detailed face, detailed hands, "
                                  "detailed feet",
                          "clip": ["2", 1]}}
    wf["4"] = {"class_type": "CLIPTextEncode",
               "_meta": {"title": "Negative"},
               "inputs": {"text": "worst quality, low quality, blurry, deformed, "
                                  "bad anatomy, bad hands, extra fingers, extra digits, "
                                  "mutated, disfigured, illustration, anime, cartoon, "
                                  "3d render, cgi, doll, plastic skin",
                          "clip": ["2", 1]}}
    wf["5"] = {"class_type": "ToBasicPipe",
               "_meta": {"title": "Basic pipe (SDXL)"},
               "inputs": {"model": ["2", 0], "clip": ["2", 1], "vae": ["2", 2],
                          "positive": ["3", 0], "negative": ["4", 0]}}

    # V107.1: 批量 -> 逐帧列表。
    # ImpactImageBatchToImageList 的 OUTPUT_IS_LIST=True，把 VHS_LoadVideo 的整批帧
    # 拆成逐帧列表；下游检测/Detailer/回贴节点均未设置 INPUT_IS_LIST，ComfyUI 会
    # 按最长输入列表逐元素迭代，从而让 SEGSDetailer 每次只处理一帧（绕开多帧 batch
    # 的 segs 对齐问题）。
    wf["6"] = {
        "class_type": "ImpactImageBatchToImageList",
        "_meta": {"title": "Batch -> Image List (for SEGSDetailer)"},
        "inputs": {"image": ["1", 0]}
    }

    prev = "6"
    nid = 100
    for i, yolo in enumerate(yolo_models):
        d = str(nid)
        wf[d] = {"class_type": "AILab_YoloV8",
                 "_meta": {"title": "Detect #%d (%s)" % (i + 1, yolo)},
                 "inputs": {"images": [prev, 0], "yolo_model": yolo, "mask_count": "2"}}
        g = str(nid + 1)
        wf[g] = {"class_type": "GrowMask",
                 "_meta": {"title": "Grow #%d" % (i + 1)},
                 "inputs": {"mask": [d, 1], "expand": 24, "tapered_corners": True}}
        s = str(nid + 2)
        wf[s] = {"class_type": "MaskToSEGS",
                 "_meta": {"title": "Mask->SEGS #%d" % (i + 1)},
                 "inputs": {"mask": [g, 0], "combined": True, "crop_factor": 1.6,
                            "bbox_fill": True, "drop_size": 10, "contour_fill": False}}
        det = str(nid + 3)
        wf[det] = {"class_type": "SEGSDetailer",
                   "_meta": {"title": "Detailer #%d (%s)" % (i + 1, yolo)},
                   "inputs": {"image": [prev, 0], "segs": [s, 0],
                              "guide_size": 384, "guide_size_for": True,
                              "max_size": 1024, "seed": 20261118, "steps": 16,
                              "cfg": 3.0, "sampler_name": "euler", "scheduler": "simple",
                              "denoise": 0.16, "noise_mask": True, "force_inpaint": True,
                              "basic_pipe": ["5", 0], "refiner_ratio": 0.2,
                              "batch_size": 1, "cycle": 1, "inpaint_model": False,
                              "noise_mask_feather": 24}}
        p = str(nid + 4)
        wf[p] = {"class_type": "SEGSPaste",
                 "_meta": {"title": "Paste #%d" % (i + 1)},
                 "inputs": {"image": [prev, 0], "segs": [det, 0],
                            "feather": 16, "alpha": 255}}
        prev = p
        nid += 10

    # V107.1: 逐帧列表 -> 重新合批，喂回 VHS_VideoCombine。
    # 本机 ComfyUI 的 /object_info 中不存在 ImpactImageListToImageBatch，
    # 因此使用内置 ImageListToImageBatch（输入键名 images，INPUT_IS_LIST=True，
    # 整体执行一次并沿 batch 维拼接）。
    wf["299"] = {
        "class_type": "ImageListToImageBatch",
        "_meta": {"title": "Image List -> Batch (for VideoCombine)"},
        "inputs": {"images": [prev, 0]}
    }
    wf.update(wf_video_combine("300", ["299", 0], prefix, FPS))
    return wf


# --------------------------------------------------------------------------- #
# Stage 4 —— 4x 光流补帧 + 60fps 重采样
# --------------------------------------------------------------------------- #
def build_wf_stage4(video_name, prefix=PREFIX + "_S4_960x1680_64fps",
                    skip_first_frames=0, frame_load_cap=0, rife_ckpt=RIFE_CKPT):
    """Stage 4: FL_RIFE 4x -> 64fps（ComfyUI），随后本地 ffmpeg -vf fps=60。"""
    wf = {}
    wf["1"] = wf_load_video(video_name, skip_first_frames=skip_first_frames,
                            frame_load_cap=frame_load_cap)
    wf["2"] = {"class_type": "FL_RIFE",
               "_meta": {"title": "RIFE 4x (16fps -> 64fps)"},
               "inputs": {"ckpt_name": rife_ckpt, "images": ["1", 0],
                          "multiplier": RIFE_MULT, "ensemble": True}}
    wf.update(wf_video_combine("3", ["2", 0], prefix, FPS * RIFE_MULT))
    return wf


def ffprobe_duration(path):
    """获取视频总时长（秒，float）。失败返回 0.0。"""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "format=duration",
                            "-of", "csv=p=0", path],
                           capture_output=True, text=True, check=True)
        return float(r.stdout.strip())
    except Exception as e:
        log("ffprobe_duration: 失败 (%s)" % e)
        return 0.0


def resample_to_60(src, dst, master_grade=False, trim_duration=-1.0, fade_out=2.0):
    """本地 ffmpeg 重采样 64fps -> 60fps，输出 960x1680@60fps。

    V108 升级：
      * master_grade 默认 False（生成端已有 LAB 0.90 锚定，不再默认叠加发灰压饱和曲线）；
      * trim_duration < 0 : 保留完整全长（不裁切 R24 尾声），默认 -1.0；
      * trim_duration == 0: 自动裁掉末尾 4 秒；
      * trim_duration > 0 : 截取前 N 秒；
      * fade_out > 0      : 末尾 N 秒黑场淡出（fade=t=out）。
    """
    which_or_die("ffmpeg")

    # ---- 计算裁剪时长 ----
    trim_req = -1.0 if trim_duration is None else float(trim_duration)
    dur = ffprobe_duration(src)
    if trim_req < 0:
        if dur <= 0:
            raise RuntimeError("无法获取 %s 的总时长" % src)
        trim = None
        eff_dur = dur
        log("Stage 4: 保留完整全长 = %.2fs (不裁剪末尾)" % eff_dur)
    elif trim_req == 0:
        if dur <= 0:
            raise RuntimeError("无法获取 %s 的总时长，--trim-duration 0 需要时长信息" % src)
        trim = max(dur - 4.0, 1.0)
        eff_dur = trim
        log("Stage 4: 自动裁剪时长 = 总时长 %.2fs - 4s = %.2fs" % (dur, trim))
    else:
        trim = trim_req
        eff_dur = trim
        log("Stage 4: 裁剪时长 = %.2fs" % trim)

    fade = float(fade_out or 0.0)
    fade_start = max(eff_dur - fade, 0.0) if fade > 0 else 0.0
    trim_args = ["-t", "%.3f" % trim] if trim is not None else []
    delivery_args = [
        "-c:v", "libx264", "-preset", "slow", "-crf", "14",
        "-profile:v", "high", "-level", "4.2", "-pix_fmt", "yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709",
        "-color_trc", "bt709", "-color_range", "tv",
        "-movflags", "+faststart", "-r", str(FINAL_FPS), dst,
    ]

    if master_grade:
        # 单次复合滤镜：分轨 -> 调色 -> 渐进混合 -> 淡出 -> 60fps
        fc = (
            "[0:v]split=2[orig][grade];"
            "[grade]curves=all='0/0.03 0.15/0.24 0.50/0.53 0.85/0.81 1/0.95',"
            "eq=saturation=0.90:contrast=0.94[fixed];"
            "[orig][fixed]blend=all_expr='A*(1-clip((T-40)/65,0,1))"
            "+B*clip((T-40)/65,0,1)'"
            + (",fade=t=out:st=%s:d=%s" % (("%.3f" % fade_start), ("%.3f" % fade))
               if fade > 0 else "")
            + ",fps=%d[out]" % FINAL_FPS
        )
        cmd = (["ffmpeg", "-y", "-v", "error", "-i", src] + trim_args +
               ["-filter_complex", fc, "-map", "[out]", "-an"] + delivery_args)
        log("Stage 4: 母片精修已启用（渐进调色 40s~105s, fade_start=%.3fs, fade=%.3fs）"
            % (fade_start, fade))
    else:
        vf = "fps=%d" % FINAL_FPS
        if fade > 0:
            vf = "fade=t=out:st=%.3f:d=%.3f,%s" % (fade_start, fade, vf)
        cmd = (["ffmpeg", "-y", "-v", "error", "-i", src] + trim_args +
               ["-vf", vf, "-an"] + delivery_args)
        log("Stage 4: 母片精修未启用（商用 BT.709 faststart 编码, fade_start=%.3fs, fade=%.3fs）"
            % (fade_start, fade))

    run(cmd)
    w, h, fps = ffprobe_size(dst)
    log("Stage 4: final -> %s (%dx%d @ %.2ffps)" % (dst, w, h, fps))
    return dst


# --------------------------------------------------------------------------- #
# 阶段调度
# --------------------------------------------------------------------------- #
def default_paths(root):
    work = os.path.join(root, "_post24R")
    return {
        "work": work,
        "s1": os.path.join(work, "stage1_merged_544x960_16fps.mp4"),
        "s2": os.path.join(work, "stage2_upscaled_960x1680_16fps.mp4"),
        "s3": os.path.join(work, "stage3_detailed_960x1680_16fps.mp4"),
        "s4_64": os.path.join(work, "stage4_rife_960x1680_64fps.mp4"),
        "final": os.path.join(work, "FINAL_960x1680_60fps.mp4"),
    }


def _output_from_history(hist, ext=".mp4"):
    """从 ComfyUI /history 响应中精确解析生成的视频文件路径。"""
    outputs = (hist or {}).get("outputs") or {}
    for _nid, node_out in outputs.items():
        if not isinstance(node_out, dict):
            continue
        for key in ("gifs", "videos", "images"):
            for item in node_out.get(key) or []:
                if not isinstance(item, dict):
                    continue
                fp = item.get("fullpath")
                if fp and fp.lower().endswith(ext) and os.path.isfile(fp):
                    return fp
                fn = item.get("filename")
                if fn and fn.lower().endswith(ext):
                    sub = item.get("subfolder") or ""
                    cand = os.path.join(COMFYUI_OUTPUT, sub, fn) if sub else os.path.join(COMFYUI_OUTPUT, fn)
                    if os.path.isfile(cand):
                        return cand
    return None


def run_comfy_stage(comfy, wf, prefix, dry_run=False):
    if dry_run:
        log("[dry-run] workflow for %s:" % prefix)
        print(json.dumps(wf, indent=2, ensure_ascii=False))
        return None
    pid = comfy.submit(wf)
    log("submitted %s (prompt_id=%s)" % (prefix, pid))
    hist = comfy.wait(pid)
    out = _output_from_history(hist) or comfy_output_latest(prefix)
    log("comfy output: %s" % out)
    return out


def main():
    global COMFY_ROOT, COMFYUI_INPUT, COMFYUI_OUTPUT
    ap = argparse.ArgumentParser(
        description="24R 四阶段后处理：接缝修复 -> 2x超分 -> 高分辨局部修复 -> 4x补帧+60fps")
    ap.add_argument("--stage", default="all",
                    help="1 | 2 | 3 | 4 | all（也接受 --stage 1 --stage 2 形式多次指定）")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--comfy-url", default=COMFY_URL_DEFAULT)
    ap.add_argument("--comfy-root", default=COMFY_ROOT, help="ComfyUI 安装父目录（含 ComfyUI/input 与 output）")
    ap.add_argument("--work-dir", default=None, help="中间产物目录（默认 <root>/_post24R）")
    ap.add_argument("--run-tag", default="", help="ComfyUI 输入/输出文件隔离标签（默认在指定 --work-dir 时自动派生）")
    ap.add_argument("--drop", type=int, default=DROP_HEAD_FROM_R2)
    ap.add_argument("--window", type=int, default=WINDOW, help="接缝软过渡窗口帧数")
    ap.add_argument("--hist-strength", type=float, default=SEAM_HIST_STRENGTH,
                    help="接缝直方图对齐强度 0~1")
    ap.add_argument("--seam-blend", dest="seam_blend", action="store_true", default=False,
                    help="启用接缝像素淡入叠加（V106 默认关闭，避免手部半透明双影）")
    ap.add_argument("--upscale-mode", choices=["model", "sdxl"], default="model")
    ap.add_argument("--sharpen", type=float, default=STAGE2_SHARPEN_AMOUNT,
                    help="Stage 2 CAS 锐化强度（默认 0.20；0=关闭）")
    ap.add_argument("--yolo-models",
                    default=r"bbox\hand_yolov8n.pt",
                    help="逗号分隔的检测器模型名（脸/手/脚），仅在 Stage 3 使用")
    ap.add_argument("--rife-ckpt", default=RIFE_CKPT)
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_DEFAULT,
                    help="Stage 2/3/4 每次向 ComfyUI 提交的分片帧数（默认 64）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印将要提交的工作流，不连 ComfyUI、不渲染")
    ap.add_argument("--max-chunks", type=int, default=0,
                    help="限制每个阶段最多处理的分片数（0=全部，>0 仅用于单切片快速测试）")
    ap.add_argument("--master-grade", dest="master_grade", action="store_true", default=False,
                    help="在 Stage 4 的 60fps 重采样中合并旧版渐进调色（40s~105s）；V108 默认关闭")
    ap.add_argument("--no-master-grade", dest="master_grade", action="store_false",
                    help="关闭母片精修（默认已关闭）")
    ap.add_argument("--trim-duration", type=float, default=-1.0,
                    help="母片截取时长（秒）；<0=保留完整全长（默认 -1），0=自动裁掉末尾 4 秒")
    ap.add_argument("--fade-out", type=float, default=2.0,
                    help="末尾黑场淡出时长（秒），默认 2.0；0 = 不淡出")
    args = ap.parse_args()

    stages = []
    for tok in str(args.stage).replace(",", " ").split():
        t = tok.strip().lower()
        if t in ("all", "*"):
            # V105: 默认跳过 Stage 3（逐帧 SDXL 重绘会引发五官闪烁）
            stages = [1, 2, 4]
            break
        if t in ("1", "2", "3", "4"):
            stages.append(int(t))
    if not stages:
        print("bad --stage %r" % args.stage)
        return 2
    stages = sorted(set(stages))

    root = os.path.abspath(args.root)
    paths = default_paths(root)
    if args.work_dir:
        paths = dict(paths)
        paths["work"] = os.path.abspath(args.work_dir)
        paths["s1"] = os.path.join(paths["work"], "stage1_merged_544x960_16fps.mp4")
        paths["s2"] = os.path.join(paths["work"], "stage2_upscaled_960x1680_16fps.mp4")
        paths["s3"] = os.path.join(paths["work"], "stage3_detailed_960x1680_16fps.mp4")
        paths["s4_64"] = os.path.join(paths["work"], "stage4_rife_960x1680_64fps.mp4")
        paths["final"] = os.path.join(paths["work"], "FINAL_960x1680_60fps.mp4")
    COMFY_ROOT = args.comfy_root
    COMFYUI_INPUT = os.path.join(COMFY_ROOT, "ComfyUI", "input")
    COMFYUI_OUTPUT = os.path.join(COMFY_ROOT, "ComfyUI", "output")
    if not args.dry_run:
        os.makedirs(paths["work"], exist_ok=True)

    run_tag = args.run_tag.strip()
    if not run_tag and args.work_dir:
        wb = os.path.basename(paths["work"].rstrip("\\/"))
        if wb and wb != "_post24R":
            run_tag = wb.lstrip("_")
    run_prefix = f"{PREFIX}_{run_tag}" if run_tag else PREFIX
    in_suffix = f"_{run_tag}" if run_tag else ""

    log("stages      : %s" % stages)
    log("root        : %s" % root)
    log("work dir    : %s" % paths["work"])
    log("run prefix  : %s" % run_prefix)
    log("comfy       : %s%s" % (args.comfy_url, "  (dry-run: 不连接)" if args.dry_run else ""))

    comfy = Comfy(args.comfy_url)
    if not args.dry_run and not comfy.alive():
        log("ERROR: ComfyUI not reachable at %s" % args.comfy_url)
        return 3

    # ---------------- Stage 1 ----------------
    if 1 in stages:
        log("=== STAGE 1: 原分辨率接缝修复与合并 (544x960@16fps) ===")
        if args.dry_run:
            est_total = ROUND_FRAMES + (ROUNDS - 1) * max(ROUND_FRAMES - args.drop, 1)
            log("[dry-run] Stage 1: 将拼接 %d 轮 (drop=%d -> 约 %d 帧) -> %s"
                % (ROUNDS, args.drop, est_total, paths["s1"]))
        else:
            stage1(root, paths["s1"], drop=args.drop, window=args.window,
                   hist_strength=args.hist_strength,
                   seam_blend=args.seam_blend)
    elif not args.dry_run and 2 in stages and not os.path.isfile(paths["s1"]):
        log("ERROR: Stage 2 需要 Stage 1 产物，缺失: %s" % paths["s1"])
        return 4

    # ---------------- Stage 2 ----------------
    if 2 in stages:
        log("=== STAGE 2: 2x 超分 (544x960 -> 960x1680 @16fps) ===")
        s1_name = f"POST24R_S1_544x960_16fps{in_suffix}.mp4"
        if args.dry_run:
            name = s1_name
            total = ROUND_FRAMES + (ROUNDS - 1) * max(ROUND_FRAMES - args.drop, 1)
        else:
            if not os.path.isfile(paths["s1"]):
                log("ERROR: 缺少 Stage 1 产物: %s" % paths["s1"])
                return 4
            name = copy_to_comfy_input(paths["s1"], s1_name)
            total = ffprobe_frames(paths["s1"])
        log("Stage 2: input frames = %d (sharpen=%.2f)" % (total, args.sharpen))

        def _b2(video_name, prefix, skip, cap):
            return build_wf_stage2(video_name, mode=args.upscale_mode, prefix=prefix,
                                   skip_first_frames=skip, frame_load_cap=cap,
                                   sharpen_amount=args.sharpen)

        out = run_chunked_stage(comfy, "S2", name, _b2,
                                os.path.join(paths["work"], "_chunks_s2"),
                                total, args.chunk_size, paths["s2"],
                                dry_run=args.dry_run, overlap=False, fps=FPS,
                                max_chunks=args.max_chunks, run_prefix=run_prefix)
        if out:
            log("Stage 2 DONE -> %s" % paths["s2"])
    elif not args.dry_run and (3 in stages or 4 in stages) and not os.path.isfile(paths["s2"]):
        log("ERROR: Stage 3/4 需要 Stage 2 产物，缺失: %s" % paths["s2"])
        return 4

    # ---------------- Stage 3 ----------------
    if 3 in stages:
        log("=== STAGE 3: 高分辨率局部修复 (960x1680 @16fps) ===")
        yolos = [y.strip() for y in args.yolo_models.split(",") if y.strip()]
        s2_name = f"POST24R_S2_960x1680_16fps{in_suffix}.mp4"
        if args.dry_run:
            name = s2_name
            total = ROUND_FRAMES + (ROUNDS - 1) * max(ROUND_FRAMES - args.drop, 1)
        else:
            if not os.path.isfile(paths["s2"]):
                log("ERROR: 缺少 Stage 2 产物: %s" % paths["s2"])
                return 4
            name = copy_to_comfy_input(paths["s2"], s2_name)
            total = ffprobe_frames(paths["s2"])
        log("Stage 3: input frames = %d" % total)

        def _b3(video_name, prefix, skip, cap):
            return build_wf_stage3(video_name, yolos, prefix=prefix,
                                   skip_first_frames=skip, frame_load_cap=cap)

        out = run_chunked_stage(comfy, "S3", name, _b3,
                                os.path.join(paths["work"], "_chunks_s3"),
                                total, args.chunk_size, paths["s3"],
                                dry_run=args.dry_run, overlap=False, fps=FPS,
                                max_chunks=args.max_chunks, run_prefix=run_prefix)
        if out:
            log("Stage 3 DONE -> %s" % paths["s3"])
    elif not args.dry_run and 4 in stages and not os.path.isfile(paths["s3"]):
        log("NOTE: 未找到 Stage 3 产物，Stage 4 将直接使用 Stage 2 的 960x1680 产物")
        if not os.path.isfile(paths["s2"]):
            log("ERROR: Stage 4 需要 Stage 2/3 产物，缺失: %s" % paths["s2"])
            return 4

    # ---------------- Stage 4 ----------------
    if 4 in stages:
        log("=== STAGE 4: 4x 光流补帧 + 60fps 重采样 ===")
        if args.dry_run:
            s4_name = f"POST24R_S2_960x1680_16fps{in_suffix}.mp4"
            name = s4_name
            total = ROUND_FRAMES + (ROUNDS - 1) * max(ROUND_FRAMES - args.drop, 1)
        else:
            # V105: 默认跳过 Stage 3 时，直接用 Stage 2 的 960x1680 产物
            if os.path.isfile(paths["s3"]):
                s4_src = paths["s3"]
                s4_name = f"POST24R_S3_960x1680_16fps{in_suffix}.mp4"
            elif os.path.isfile(paths["s2"]):
                s4_src = paths["s2"]
                s4_name = f"POST24R_S2_960x1680_16fps{in_suffix}.mp4"
                log("Stage 4: 使用 Stage 2 产物作为输入（跳过 Stage 3）: %s" % s4_src)
            else:
                log("ERROR: 缺少 Stage 2/3 产物，无法执行 Stage 4")
                return 4
            name = copy_to_comfy_input(s4_src, s4_name)
            total = ffprobe_frames(s4_src)
        log("Stage 4: input frames = %d" % total)

        def _b4(video_name, prefix, skip, cap):
            return build_wf_stage4(video_name, prefix=prefix,
                                   skip_first_frames=skip, frame_load_cap=cap,
                                   rife_ckpt=args.rife_ckpt)

        out = run_chunked_stage(comfy, "S4", name, _b4,
                                os.path.join(paths["work"], "_chunks_s4"),
                                total, args.chunk_size, paths["s4_64"],
                                dry_run=args.dry_run, overlap=True,
                                fps=FPS * RIFE_MULT, max_chunks=args.max_chunks,
                                run_prefix=run_prefix)
        if out:
            resample_to_60(paths["s4_64"], paths["final"],
                           master_grade=args.master_grade,
                           trim_duration=args.trim_duration,
                           fade_out=args.fade_out)
            log("Stage 4 DONE -> %s" % paths["final"])

    log("ALL DONE. 产物:")
    for k in ("s1", "s2", "s3", "s4_64", "final"):
        if os.path.isfile(paths[k]):
            log("  %-7s %s" % (k, paths[k]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
