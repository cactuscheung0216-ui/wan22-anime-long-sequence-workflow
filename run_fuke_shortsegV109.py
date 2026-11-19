# -*- coding: utf-8 -*-
r"""
FUKE_ShortSegV109 - Wan 2.2 I2V-A14B 24-Round x ~5s Full-Body Pipeline (544x960 @ 16fps)

Architecture & Key Upgrades (V109):
  1) P1 Golden Sampling Kernel (restored & verified across 24 rounds):
     - steps 2 + 4 + 4 = 10, CFG = 3.0, scheduler = simple, sampler = euler
     - motion = 1.18, latent_refresh = 0.80, MLC = 1, lock_start_slots = 1
     - add_reference_latents = False (prevents base-image silhouette/halo ghosting & background grid)
     - per-round LAB ImageColorMatch+ factor = 0.90 (display/video output branch only; raw latent preserved)
  2) Full-Body Anti-Crop Framing & Motion Guardrails (V108/V109):
     - 9:16 portrait framing (544x960, 81 frames/round @ 16fps = 5.06s; drop=4 from R02+ -> 1,852 frames)
     - Full-length silhouette + visible heels anchors in OUTFIT_LOCK / DANCE_BASE + anti-crop negatives
     - MOTION_GUARDRAIL without slow-word conflicts against FORBIDDEN_ALWAYS
  3) Reliability & Proxy Isolation:
     - Loopback proxy bypass via urllib.request.ProxyHandler({})
     - Strict 24-round workflow/prompt --dry-run validation (zero side effects)
     - Automatic cv2/numpy pixel-pop teleport QC with seed retry

Environment Variables (optional overrides):
  COMFY_ROOT       ComfyUI installation parent directory (default: C:\ComfyUI)
  COMFY_URL        ComfyUI HTTP base URL (default: http://127.0.0.1:8188)
  FUKE_OUTPUT_DIR  Output directory for rounds (default: ./output_fuke_shortsegV109_P1)
  ORIGINAL_BASE    Anchor image filename in <ComfyUI>/input/ (default: your_anchor_image.png)
  PY_COMFY         Optional Python executable path containing cv2 + numpy for QC

Usage:
  python run_fuke_shortsegV109.py --dry-run
  python run_fuke_shortsegV109.py --rounds 6 --base-image your_anchor.png
  python run_fuke_shortsegV109.py --comfy-root "D:\ComfyUI" --base-image your_anchor.png
  python run_fuke_shortsegV109.py --start-round 10 --rounds 24
  Post-process 24 rounds to commercial 960x1680 @ 60fps BT.709:
  python postprocess_24R_V109.py --stage all --root ./output_fuke_shortsegV109_P1
"""
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time

try:
    import cv2
    import numpy as np
    HAVE_CV = True
except ImportError:
    HAVE_CV = False

COMFY_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
API_URL = f"{COMFY_URL}/prompt"
HISTORY_URL = f"{COMFY_URL}/history"
COMFY_ROOT = os.environ.get("COMFY_ROOT", r"C:\ComfyUI")
COMFYUI_INPUT = os.path.join(COMFY_ROOT, "ComfyUI", "input")
COMFYUI_OUTPUT = os.path.join(COMFY_ROOT, "ComfyUI", "output")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _build_python_candidates(comfy_root):
    cands = [
        os.environ.get("PY_COMFY", ""),
        sys.executable,
        os.path.join(comfy_root, "python", "python.exe"),
        os.path.join(comfy_root, "python_embeded", "python.exe"),
        os.path.join(comfy_root, "venv", "Scripts", "python.exe"),
        os.path.join(comfy_root, ".venv", "bin", "python"),
        shutil.which("python") or "",
    ]
    return [c for c in cands if c]


# 候选解释器：自动切到带 cv2/numpy 的 Python（否则自动质检会被静默关闭）
PYTHON_CANDIDATES = _build_python_candidates(COMFY_ROOT)
REEXEC_GUARD = "FUKE_SHORTSEG_V109_REEXEC"

BOOTSTRAP_WORKFLOW_FILE = os.path.join(SCRIPT_DIR, "workflow_fuke_shortsegV109_bootstrap.json")
CONTINUE_WORKFLOW_FILE = os.path.join(SCRIPT_DIR, "workflow_fuke_shortsegV109_continue.json")

OUTPUT_DIR = os.environ.get("FUKE_OUTPUT_DIR", os.path.join(SCRIPT_DIR, "output_fuke_shortsegV109_P1"))

TEST_MODE = False
TEST_ROUNDS = 6
TOTAL_ROUNDS = 24         # 24 轮，每轮 81 帧 @ 16FPS = 5.06 秒
FRAMES_PER_ROUND = 81     # 每轮帧数
RESOLUTION_W = 544
RESOLUTION_H = 960
OUTPUT_PREFIX = "FUKE_ShortSegV109_P1"
LATENT_SAVE_DIR = "fuke_shortsegV109_P1_latent"
ORIGINAL_BASE = os.environ.get("ORIGINAL_BASE", "your_anchor_image.png")
SEED_BASE = 123456789

CM_FACTOR = 0.90
LATENT_REFRESH = 0.80       # V109: P1 黄金内核 0.80（压制高频噪点/对比度漂移）
MOTION = 1.18               # V109: P1 黄金内核 1.18（配合 MLC=1 与 10 步采样）
MOTION_BOOTSTRAP = 1.18
CFG = 3.0
DROP_HEAD_FROM_R2 = 4       # V109: 匹配 MLC=1 / lock_start_slots=1 (R02~R24 砍头 4 帧，保留对齐帧 idx=4)
FPS = 16
MLC_BOOTSTRAP = 1
MLC_CONTINUE = 1            # V109: P1 黄金内核 MLC=1，避免双 slot 锁定带来的误差与接缝偏移

LATENT_INPUT_NAME = "fuke_shortsegV109_P1_prev.latent"
MAX_RETRIES = 2
# ---- 自动质检(传送检测) ----
QC_RETRIES = 2
POP_FRAC = 7.5     # 该步大变化像素占比(%) 下限
POP_RATIO = 4.0    # 该步占比 / 周围中位数 下限
SEED_RETRY_STEP = 7919
# ---- 续跑来源（默认=本版自己的输出；要接别的链请用 --prev-dir 并改 PREV_PREFIX）----
PREV_PREFIX = OUTPUT_PREFIX
PREV_DIR_DEFAULT = OUTPUT_DIR
ROUND_TIMEOUT_MIN = 90
MIN_ROUND_SECONDS = 20   # 采样 14B x 81 帧不可能 <20s 完成；更快一律视为缓存/假成功，换种子重跑


# ---------------------------------------------------------------- 提示词层 V109
#
# 默认示例针对「日光花园丝质旗袍全身立姿」锚点底图编写；若更换角色/场景底图，
# 请同步替换 OUTFIT_LOCK、SCENE_LOCK 与 DANCE_BASE 中对应视觉描述，并保持四段式结构与断言。
# 构图约定：9:16 竖屏 544x960 全身入画（头到高跟鞋完整保留），避免退化为仅上半身挥手。

OUTFIT_LOCK = (
    "same character identity as reference image, photorealistic real woman, "
    "young Asian woman with black hair tied in a low side ponytail and loose floating strands around her face, "
    "warm brown eyes, gentle smile, parted lips, natural makeup with light red lipstick, "
    "slender figure, delicate features, "
    "red silk qipao cheongsam with jade-green and golden-yellow peony embroidery, "
    "mandarin collar, frog buttons, side slits, sheer short sleeves, "
    "full-length figure from head to toe inside the frame, "
    "both matching red heels always visible on the garden ground, jade pendant, "
    "no costume change, no hairstyle change, no twin tails"
)

# SCENE_LOCK：与 V103 同一语义（相机锁死 + 背景静止 + 环境动），措辞按新底图改写
# （静态机位 + 锁定背景景物，仅微风与斑驳阳光保持连续动态）
SCENE_LOCK = (
    "static camera, locked-off eye-level shot, "
    "the red and pink peonies, the green bamboo and the blue sky stay perfectly still, "
    "the light breeze and the dappled sunlight keep moving"
)

# ---------------------------------------------------------------- 动作底座
# V102: "anime girl performing a professional Korean pop dance choreography, ..."
# V103: 底图是写实照片 —— "anime girl" 会把脸和皮肤拉回插画风，与底图打架
DANCE_BASE = (
    "photorealistic woman standing in a sunlit garden among red and pink peonies and green bamboo, "
    "full-body framing keeping her entire silhouette and red heels well inside the frame, "
    "her whole body committed to the movement, "
    "calm poised presence, gentle warm expression, "
    "her legs and hips carrying her weight with grounded weight shifts, "
    "the light breeze and the dappled sunlight shaping her hair and the sheen of her silk qipao, "
    "one continuous action unfolding without a cut"
)

# 24 轮的类别（与 V92 / V102 / V103 完全相同，保证逐档对比成立）
# 段落: 序 R01-03 | 主歌 R04-08 | 预副歌 R09-11 | 副歌1 R12-15 | 间奏 R16-17 | 副歌2 R18-21 | 尾声 R22-24
ROUND_CLASS = "MMQMQSMQMQSPQMPSMPSQPMPS"

# 手势母题（原文回归，搭配不同的动作）—— 与 V92 / V102 / V103 相同
MOTIF_A = "her right hand draws a line from her hip up to her cheek and pushes out toward the camera"
MOTIF_B = "her arms cross in front of her chest and snap open wide"
MOTIF_A_ROUNDS = (0, 11, 22)    # R01 / R12 / R23
MOTIF_B_ROUNDS = (4, 14, 20)    # R05 / R15 / R21

VARIATIONS = [
    # R01 M 序（母题 A）—— 底图是静站：用左右 step touch 把「站住」变成「站中动起来」
    "in an even mid-tempo, her heels step touch from side to side on the garden ground with her weight rolling onto each leg and her level dipping slightly on every touch, "
    "her right hand draws a line from her hip up to her cheek and pushes out toward the camera while her shoulders hit the beat twice, "
    "a warm confident gaze held on the camera",
    # R02 M 序
    "in an even mid-tempo, she cross steps forward with one heel crossing over the other and her weight travelling with the step, "
    "her torso twisting a beat later to follow while her arms swing out wide and draw back in, "
    "her head tilting with a warm sideways glance back to the camera",
    # R03 Q 序
    "on a quicker phrase, her heels taking quick double steps on the beat as her weight springs from one heel to the other, "
    "her chest popping on the second step while her hips snap back to centre and her arms swing wide, "
    "her eyes locked on the camera",
    # R04 M 主歌
    "in an even mid-tempo, she steps wide to the right and drops her weight into that leg so her hips sink and rebound, "
    "her torso twisting with the step while her left arm sweeps across her body and her right hand rises to her hair, "
    "her gaze following her hand before it returns to the camera",
    # R05 Q 主歌（母题 B）
    "on a quicker phrase, her heels doing quick step touches to the left and right as her weight shifts hard between them, "
    "her arms cross in front of her chest and snap open wide with her shoulders popping twice on the beat, "
    "her eyes locked on the camera",
    # R06 S 主歌
    "on a slower phrase, she bends her knees to lower her level slightly while keeping both heels planted inside the frame, "
    "her chest opening wide as her arms sweep out and her torso arches back, "
    "then she pushes back up with her chin lifting and her eyes on the camera",
    # R07 M 主歌
    "in an even mid-tempo, her hips drawing a figure eight while her heels stay grounded on the garden ground and her weight rocks across both legs, "
    "both hands sliding from her waist to her ribs with her torso rolling a beat later behind them, "
    "her head turning to look over her shoulder toward the camera",
    # R08 Q 主歌（身体转身，必须回镜头）
    "on a quicker phrase, she pivots a quarter to the right on her heel with her hips leading the turn and her weight driving it, "
    "her arms leading the turn and her shoulders swinging through it, "
    "then she snaps back to face the camera and drops her hips into a low stance with her eyes locked on it",
    # R09 M 预副歌
    "in an even mid-tempo, her weight shifting left and right as her heels lift and drop on the garden ground to tap out the rhythm, "
    "her shoulders counter-rotating while both hands sweep outward beside her face and press away from the lens, "
    "her eyes narrowing with intensity at the camera",
    # R10 Q 预副歌
    "on a quicker phrase, her knees dipping and lifting on every beat as her weight pulses from one leg to the other, "
    "both arms punching out to the sides in alternation with her chest popping and her shoulders driving the beat, "
    "her head snapping toward the camera",
    # R11 S 预副歌（蓄力）
    "on a slower phrase, she sinks slightly into a controlled knee bend that drops her hips and opens her stance wide, "
    "her arms sinking low with the same curve and her waist folding, "
    "then she rises back up on straightened legs and her gaze lifting to the camera",
    # R12 P 副歌1（母题 A）
    "on a quicker phrase with powerful sharp accents, her weight driving down hard into her left hip as her heels plant wide and her knees press deep, "
    "her right hand draws a line from her hip up to her cheek and pushes out toward the camera while her torso snaps upright with three sharp accents and her chest lifting on every accent, "
    "an intense gaze at the camera",
    # R13 Q 副歌1
    "on a quicker phrase, she travels two wide steps to the left with her weight dropping into every landing and her knees bending on each step, "
    "her arms crossing and opening on every beat with her shoulders hitting the accents, "
    "her eyes on the camera",
    # R14 M 副歌1（身体转身，必须回镜头）
    "in an even mid-tempo, she turns a half turn to the left with her weight driving the turn and her heels pivoting under her, "
    "her arms trailing wide and her waist twisting after the turn, "
    "then she turns back to face the camera with her weight landing on her front leg and one hand at her hip for a confident half smile",
    # R15 P 副歌1（母题 B）—— floor -> garden ground
    "on a quicker phrase with powerful sharp accents, her weight driving her hips hard to the right as her heels push off the garden ground and her knees bend deep, "
    "her arms cross in front of her chest and snap open wide as she pushes up tall with her shoulders driving, "
    "then both arms drive out to the side on the beat with an intense gaze at the camera",
    # R16 S 间奏
    "on a slower phrase, she sinks into a low stance with her hips pushed to one side and her weight leaning over that leg, "
    "one arm sweeping in a long arc along her side with her torso held over the bent leg, "
    "her head turned in profile with her eyes half closed",
    # R17 M 间奏
    "in an even mid-tempo, she pushes up from the low stance on straightened legs and shifts her weight to the left so her hips follow a beat later, "
    "both hands sliding along her waist with her torso twisting after the hips, "
    "then her gaze turns back to the camera",
    # R18 P 副歌2
    "on a quicker phrase with powerful sharp accents, she travels three wide steps to the right with her weight dropping into every step and her knees bending on each landing, "
    "a sharp shoulder hit on each landing with one arm slicing across her body, "
    "then she drops into a strong open stance with both arms thrown wide and an intense gaze",
    # R19 S 副歌2
    "on a slower phrase, she stands tall in a strong stance with her weight on one hip and her front heel pointed forward, "
    "one hand gliding up from her waist along her neck with her chest open, "
    "her head tilted back and her eyes closing",
    # R20 Q 副歌2
    "on a quicker phrase, her weight pulsing from one leg to the other with her knees dipping in time and her hips stacking on each beat, "
    "her arms alternately thrusting up and out to the sides with her waist snapping on every change, "
    "her head snapping toward the camera",
    # R21 P 副歌2（母题 B）
    "on a quicker phrase with powerful sharp accents, her weight driving her down into a controlled bend with her heels planted wide and her knees folding under her, "
    "her arms cross in front of her chest and snap open wide as she rises and drops again on the beat, "
    "her hips driving up and her chest opening on every drop with an intense gaze at the camera",
    # R22 M 尾声
    "in an even mid-tempo, she pushes up from the low stance with her weight rolling onto her front leg and her heels stepping forward, "
    "her arms opening wide on the rise with one hand sweeping across her chest and her torso twisting to the left, "
    "her eyes on the camera",
    # R23 P 尾声（母题 A 双手版）
    "on a quicker phrase with powerful sharp accents, her weight dropping into her hips with her heels wide and her knees bending as both hands draw a line from her hips up to their cheeks and push out toward the camera, "
    "then she throws her arms wide with a sharp shoulder hit on every beat and her chest opening, "
    "her chin lifted and her eyes locked on the camera",
    # R24 S 尾声（收束）—— 收回到「静立 + 微笑」气质（与底图的 gentle smile 一致）
    "on a slower phrase, the movement resolves as she steps back and sinks slightly into a poised finish with her hips still rolling and both heels planted inside the frame, "
    "both arms open toward the camera with her torso coming up to meet them, "
    "her gaze held on the lens with a very faint smile",
]

TEMPO_WORD = {"M": "mid-tempo", "Q": "quicker", "P": "quicker", "S": "slower"}

# ---------------------------------------------------------------- 禁词表 V103
# 与 V102 逐条相同（结构与规范不变）：
#  - 解禁 heel / toe / foot / footwork（全身底图后这些是写腿部动作的必要词汇）
#    V106: 24 轮动作层的用词已统一为 heels（底图是红色高跟鞋）
#  - 仍禁 kick / stomp / leap / jump / spin / twirl
#    （旗袍高开衩 + 大幅踢腿 = 大腿暴露；这组词也是 G3 抖动的来源）
FORBIDDEN_ALWAYS = (
    "leap", "jump", "spin", "twirl", "whip", "fling", "explosive", "sprint", "run",
    "huge", "giant", "massive", "violent", "shake", "rapid", "bounce",
    "across the frame", "full-body", "dynamic freeze",
    "slowly", "gracefully", "gently", "delicately", "languid", "sluggish",
    "lazy", "dawdling", "freeze", "frozen", "posing", "stiff", "sliding into",
    "hold a pose", "in a pose", "natural", "smooth",
    "kick", "stomp",
)
# 幅度限制词 / 娇柔词: 除 S 轮外禁用
FORBIDDEN_NON_SLOW = (
    "small", "short", "light", "in place", "calm", "settle", "rest", "elbows held close",
    "cheerful", "playful", "cute", "bright smile", "coy", "shy",
)
# 复合词整串禁用（词边界匹配漏掉 kicking / jumping / running 这类）
FORBIDDEN_SUBSTRING = (
    "kicking", "kicks", "jumping", "spinning", "running", "sprinting", "leaping",
    "twirling", "whipping",
)
# 供词表审计使用的禁词表（词边界匹配）
FORBIDDEN_IN_VARIATION = FORBIDDEN_ALWAYS

# ---------------------------------------------------------------- V103 身体分区词表
# 与 V102 同一套词表（结构与规范不变）
LEG_WORDS = (
    "feet", "foot", "legs", "leg", "knee", "knees", "step", "steps", "stepping",
    "pivot", "pivots", "pivoting", "heel", "heels", "toe", "toes",
    "planted", "push off", "stride",
)
LOWER_BODY_WORDS = (
    "hips", "hip", "waist", "pelvis", "thigh", "weight", "level", "stance",
    "knees", "knee", "bend", "bends", "sinks", "sink", "rise", "rises", "pushes up",
    "crouch", "step", "steps", "rock", "roll", "rolls", "drop", "drops", "lift",
    "lifts", "lean", "sway", "travel", "drive", "shift", "shifts", "pulse",
    "pulses", "stack", "tilt", "tilts", "angle", "angled", "rebound", "swivel",
    "punch", "dip", "dips", "pivot", "pivots", "planted", "feet", "foot", "legs",
)
TORSO_WORDS = (
    "torso", "chest", "shoulder", "shoulders", "ribs", "spine", "waist",
)
MOMENTUM_WORDS = (
    "weight", "level", "stance", "sink", "sinks", "sinking", "rise", "rises",
    "rising", "drop", "drops", "dropping", "pushes up", "push up", "bend",
    "rock", "rocks", "roll", "rolls", "punch", "swivel", "travel", "step",
    "steps", "shift", "shifts", "lean", "rebound", "pivot", "pivots", "dip",
    "dips", "spring", "springs",
)
ARM_WORDS = ("arm", "arms", "hand", "hands", "elbow", "wrist")
HEAD_WORDS = (
    "head", "chin", "gaze", "eyes", "eye", "look", "hair", "expression",
    "profile", "camera", "lens",
)
CAMERA_CUE_WORDS = (
    "camera", "lens", "gaze", "eyes", "looks", "look ", "smile", "wink",
    "grin", "expression",
)
BODY_TURN_PHRASES = (
    "turns a quarter", "turns a half", "turns her torso", "turns back",
    "turning her torso", "turning a quarter", "her hips turn",
    "half turn", "quarter turn", "turns to face", "turns around",
    "pivot", "pivots", "pivoting",
)
TURN_CLOSURE_PHRASES = (
    "back around", "facing the camera again", "comes back around",
    "stops facing front", "turns back to face", "facing front",
    "facing the camera", "face the camera", "faces the camera",
    "front again", "back to the camera", "snaps back to face",
)

MOTION_PROMPTS = [f"{DANCE_BASE}, {v}" for v in VARIATIONS]

# ---------------------------------------------------------------- V108 护栏常量
# V108: 移除与 FORBIDDEN_ALWAYS 冲突的慢速柔弱词（slowly/gracefully/gently），
#       保留重心连续与防遮脸护栏，彻底解除 R01 静帧惰性
MOTION_GUARDRAIL = (
    ", continuous natural weight shift, "
    "hands away from face, unobstructed face"
)
# V106: 负向护栏（追加在 NEGATIVE 末尾）
NEGATIVE_GUARDRAIL = (
    ", fast motion, motion blur, waving hands rapidly, "
    "hands covering face, crossed arms over face"
)

# NEGATIVE = V102 原版（V92 的 63 条 + V102 的 20 条全部保留，0 删除）
#           + V103 追加的「反插画风 / 反失真」一组 —— 底图是写实照片，
#           而旧提示词的 anime girl 会把脸拉回插画风，这里用负面词补一道保险。
NEGATIVE = (
    "blurry, low quality, distorted face, extra limbs, watermark, text, "
    "letters, words, signature, logo, subtitles, caption, typography, "
    "written characters, Chinese characters, English text, glyphs, "
    "pale washed out colors, desaturated, grayish, "
    "different outfit, costume change, clothing change, hair color change, "
    "twin tails, high ponytail, short bob haircut, blunt bangs, "
    "short hair, different dress, western dress, plain cheongsam, "
    "barefoot, barefoot sandals, flat shoes, sneakers, white shoes, "
    "night scene, indoor scene, studio backdrop, cold blue light, overcast sky, "
    "extra hair ornaments changing every frame, "
    "camera shake, camera movement, handheld, zoom in, zoom out, "
    # 场景防抖/防背景运动项（与花园场景对齐）
    "shaking background, moving bamboo, bamboo leaves swaying, peony branches moving, "
    "background motion, jitter, flicker, motion blur, "
    # G3f 追加：反摆 pose
    # 站立微动允许脚踝重心调整，反静态项聚焦无躯干/全身僵死
    #           "完全不动" 的说法，避免把脚踝呼吸感一起压死
    "sluggish movement, lazy motion, frozen pose, stiff pose, "
    "holding a pose, posing for the camera, dead still body, no body movement, "
    # G3f 追加：反抖动
    "violent shaking, frame jitter, screen shake, excessive motion blur, "
    "whip pan, violent hair whip, distorted motion, "
    # G3g 追加：反娇柔
    "cutesy, coy pose, shy timid posture, bashful, "
    # V102 追加：反"只动上半身 / 只摆手 / 髋锁死 / 腿脚不动 / 无重心漂浮"
    "upper body only movement, upper body movement only, "
    "hand gestures alone, gesturing with the hands only, "
    "static lower body, static legs, motionless feet, legs held still, "
    "locked hips, frozen hips, stiff hips, "
    "weightless floating, floating without weight, "
    "dance pose without body travel, torso only dancing, "
    "head and arms only, no body weight shift, "
    "kicking the leg up, exposed thigh, bare thigh, "
    # V108 追加：防脚部与高跟鞋出画
    "cropped feet, feet out of frame, cut off legs, cut off ankles, "
    "close-up, medium shot, zooming in, "
    # V103 追加：反插画风 / 反失真（底图是写实照片）
    "anime, anime style, cartoon, illustration, drawing, painting, "
    "3d render, cel shading, flat color, chibi, doll, "
    "plastic skin, waxy skin, airbrushed skin, oversmoothed skin, "
    "doll face, oversized eyes, doll-like face"
)


# ---------------------------------------------------------------- 运行基础设施
def _find_python_with_cv():
    """找一个装了 cv2/numpy 的解释器；找不到返回 None"""
    for exe in PYTHON_CANDIDATES:
        if not exe or not os.path.exists(exe):
            continue
        try:
            r = subprocess.run([exe, "-c", "import cv2, numpy"], capture_output=True, timeout=60)
            if r.returncode == 0:
                return exe
        except Exception:
            continue
    return None


def ensure_qc_python():
    """若当前解释器没有 cv2，自动用带 cv2 的解释器重启自己（一次）。

    没有这一步，用默认 python 跑会静默关闭自动质检（单步传送检测），
    这正是 G3f 想保留的核心保险。
    """
    global HAVE_CV
    if HAVE_CV or os.environ.get(REEXEC_GUARD):
        return
    exe = _find_python_with_cv()
    if not exe:
        print("[QC] 未找到带 cv2/numpy 的 Python，自动质检不可用（保持关闭）")
        return
    if os.path.normcase(os.path.abspath(sys.executable)) == os.path.normcase(os.path.abspath(exe)):
        return
    env = dict(os.environ)
    env[REEXEC_GUARD] = "1"
    print(f"[QC] 切换到带 cv2 的解释器: {exe}")
    try:
        r = subprocess.run([exe, os.path.abspath(__file__)] + sys.argv[1:], env=env)
    except Exception as e:
        print(f"[QC] 重启失败: {e}")
        return
    sys.exit(r.returncode)


def load_workflow(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def ensure_base_image_in_input():
    """确保底图已在 ComfyUI input 目录就绪；若仅存在于 output 则自动复制一份。"""
    dst = os.path.join(COMFYUI_INPUT, ORIGINAL_BASE)
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        print(f"[BASE] input 已就绪: {dst}")
        return
    fallback_src = os.path.join(COMFYUI_OUTPUT, ORIGINAL_BASE)
    if os.path.exists(fallback_src) and os.path.getsize(fallback_src) > 1000:
        os.makedirs(COMFYUI_INPUT, exist_ok=True)
        shutil.copy2(fallback_src, dst)
        print(f"[BASE] {fallback_src} -> {dst} ({os.path.getsize(dst)} bytes)")
        return
    raise RuntimeError(
        f"底图缺失: 请将参考图 {ORIGINAL_BASE} 放入 {COMFYUI_INPUT} "
        f"或通过 --base-image / ORIGINAL_BASE 指定文件名"
    )


def build_round_workflow(round_idx, seed_offset=0):
    import re as _re
    use_bootstrap = (round_idx == 0)
    path = BOOTSTRAP_WORKFLOW_FILE if use_bootstrap else CONTINUE_WORKFLOW_FILE
    wf = load_workflow(path)

    motion = MOTION_PROMPTS[round_idx % len(MOTION_PROMPTS)]
    prompt = f"{motion}, {SCENE_LOCK}, {OUTFIT_LOCK}{MOTION_GUARDRAIL}"
    # V107: 轮次类别 / 动作文本（供 motion 与 latent_refresh 动态调整使用）
    cls = ROUND_CLASS[round_idx % len(ROUND_CLASS)]
    var = motion.replace(DANCE_BASE, "")
    wf["50"]["inputs"]["text"] = prompt
    wf["51"]["inputs"]["text"] = NEGATIVE + NEGATIVE_GUARDRAIL

    round_label = f"R{round_idx+1:02d}"
    wf["160"]["inputs"]["filename_prefix"] = f"{OUTPUT_PREFIX}_{round_label}"
    if use_bootstrap:
        wf["181"]["inputs"]["filename_prefix"] = f"{LATENT_SAVE_DIR}/{OUTPUT_PREFIX}_R01_FINAL"
    else:
        wf["180"]["inputs"]["latent"] = LATENT_INPUT_NAME
        wf["181"]["inputs"]["filename_prefix"] = f"{LATENT_SAVE_DIR}/{OUTPUT_PREFIX}_{round_label}_FINAL"

    seed = SEED_BASE + round_idx * 1000 + seed_offset
    if "40" in wf:
        if "seed" in wf["40"]["inputs"]:
            wf["40"]["inputs"]["seed"] = seed
        elif "value" in wf["40"]["inputs"]:
            wf["40"]["inputs"]["value"] = seed

    # V109: 恢复 P1 已全量验证的黄金采样内核 (steps 2+4+4=10, MLC=1, add_reference_latents=False, lr=0.80, motion=1.18)
    round_motion = MOTION_BOOTSTRAP if use_bootstrap else MOTION
    round_lr = LATENT_REFRESH
    wf["100"]["inputs"]["latent_refresh"] = round_lr
    wf["100"]["inputs"]["motion"] = round_motion
    wf["33"]["inputs"]["value"] = CFG
    wf["13"]["inputs"]["image"] = ORIGINAL_BASE
    wf["150"]["inputs"]["factor"] = CM_FACTOR
    wf["150"]["inputs"]["color_space"] = "LAB"

    # ---------------- asserts: G 结构 + G3f 文案 ----------------
    assert wf["181"]["class_type"] == "SaveLatent"
    assert wf["181"]["inputs"]["samples"] == ["103", 0]
    assert "170" not in wf and "171" not in wf
    assert "302" not in wf and "303" not in wf  # no IPAdapter
    assert wf["150"]["class_type"] == "ImageColorMatch+"
    assert abs(float(wf["150"]["inputs"]["factor"]) - CM_FACTOR) < 1e-6
    assert wf["150"]["inputs"]["reference"] == ["145", 0]
    assert wf["145"]["inputs"]["samples"] == ["17", 0]
    assert wf["100"]["inputs"]["anchor_samples"] == ["17", 0]
    assert wf["100"]["inputs"].get("add_reference_latents") is False
    assert int(wf["100"]["inputs"].get("lock_start_slots", 0)) == 1
    assert abs(float(wf["100"]["inputs"]["latent_refresh"]) - round_lr) < 1e-6
    assert abs(float(wf["100"]["inputs"]["motion"]) - round_motion) < 1e-6
    assert wf["160"]["inputs"]["images"] == ["150", 0]
    # 生成参数与 P1 黄金内核一致 (steps 2+4+4=10)
    assert wf["32"]["inputs"]["value"] == 10
    assert (wf["101"]["inputs"]["end_at_step"], wf["102"]["inputs"]["end_at_step"]) == (2, 6)
    assert (wf["103"]["inputs"]["start_at_step"], wf["103"]["inputs"]["end_at_step"]) == (6, 10)
    assert wf["13"]["inputs"]["image"] == ORIGINAL_BASE
    # V109 关键参数：分辨率 544x960 / 帧数 81 / 帧率 16
    assert int(wf["15"]["inputs"]["desired_width"]) == RESOLUTION_W
    assert int(wf["15"]["inputs"]["desired_height"]) == RESOLUTION_H
    assert int(wf["100"]["inputs"]["length"]) == FRAMES_PER_ROUND
    assert int(wf["160"]["inputs"]["frame_rate"]) == FPS
    assert round(FRAMES_PER_ROUND / FPS, 2) == 5.06
    # 文案：动作在前、场景锁、造型锁、负面词
    assert prompt.startswith(motion)
    assert prompt.endswith(MOTION_GUARDRAIL)
    assert MOTION_GUARDRAIL in prompt
    assert NEGATIVE_GUARDRAIL in wf["51"]["inputs"]["text"]
    assert SCENE_LOCK in prompt
    assert OUTFIT_LOCK in prompt
    assert "red silk qipao cheongsam" in prompt
    assert "black hair tied in a low side ponytail" in prompt
    assert "matching red heels" in prompt
    assert "jade-green and golden-yellow peony embroidery" in prompt
    assert motion.startswith(DANCE_BASE)
    # G3g: 按轮次类别检查词表
    for _w in FORBIDDEN_ALWAYS:
        assert not _re.search(r"\b" + _re.escape(_w) + r"\b", var, _re.IGNORECASE), \
            (f"forbidden word in variation: {_w}", var)
    for _w in FORBIDDEN_SUBSTRING:
        assert _w not in var.lower(), (f"forbidden compound in variation: {_w}", var)
    if cls != "S":
        for _w in FORBIDDEN_NON_SLOW:
            assert not _re.search(r"\b" + _re.escape(_w) + r"\b", var, _re.IGNORECASE), \
                (f"forbidden word in variation: {_w}", var)
    assert ("powerful" in var) == (cls == "P"), var
    assert TEMPO_WORD[cls] in var, var
    # V103: 母题必须在指定轮次原文出现，且不出现在别处（单镜版母题集为空）
    ri24 = round_idx % len(ROUND_CLASS)
    assert (("a line from her hip" in var) == (ri24 in MOTIF_A_ROUNDS)), var
    assert ((MOTIF_B in var) == (ri24 in MOTIF_B_ROUNDS)), var
    assert cls != "P" or ri24 in (11, 14, 17, 20, 22)
    # G3f 关键 2：SCENE_LOCK / NEGATIVE 与 G3e 基线一致（相机锁死）
    assert "stay perfectly still" in SCENE_LOCK
    assert "the light breeze and the dappled sunlight keep moving" in SCENE_LOCK
    assert "camera shake" in NEGATIVE
    assert "letters" in NEGATIVE and "signature" in NEGATIVE
    # G3f 关键 3：负面词必须同时含"反摆 pose"与"反抖动"
    assert "holding a pose" in NEGATIVE and "dead still body" in NEGATIVE
    assert "coy pose" in NEGATIVE
    assert "frame jitter" in NEGATIVE and "violent shaking" in NEGATIVE
    # V92 关键 4：快慢必须成对出现（既有 quicker 也有 slower），不能全片一种速度
    assert ("quicker" in motion) or ("slower" in motion) or ("mid-tempo" in motion)
    # V103 关键 4（核心：腿部轴）—— 每个动作单元段都必须真的写腿部动作。
    #   V92 的失败形态是「手臂/手/头/上半身动作」，所以这一条是本版的存在理由。
    assert any(w in var for w in LEG_WORDS), ("no leg action", motion)
    # V103 关键 5（下半身/重心轴）—— 腰/髋/骨盆/站距/重心/身体高度
    assert any(w in var for w in LOWER_BODY_WORDS), ("no lower-body axis", motion)
    # V103 关键 6（重心轴）—— 重心或身体高度必须真的在变
    assert any(w in var for w in MOMENTUM_WORDS), ("no weight/level change", motion)
    # V103 关键 7（躯干轴）—— 不是只有腿在动，上身要有配合
    assert any(w in var for w in TORSO_WORDS), ("no torso axis", motion)
    # V103 关键 8（手臂轴）—— 保留 V92 的表现力，手臂不能丢
    assert any(w in var for w in ARM_WORDS), ("no arm axis", motion)
    # V103 关键 9（对镜头表演）—— 每轮都要有对镜头的表演线索
    assert any(w in var for w in CAMERA_CUE_WORDS), ("no camera-facing cue", motion)
    # V103 关键 10（转身闭合）—— 含身体旋转的轮次必须回到面向镜头
    if any(t in var for t in BODY_TURN_PHRASES):
        assert any(w in var for w in TURN_CLOSURE_PHRASES), ("turn without returning to camera", motion)
    # V103 关键 11（动作单元上限）—— 单镜 5.06 秒内人类可完成 3 个连贯动作。
    #   节奏标签不算动作单元。
    units = [u for u in var.split(",") if u.strip()]
    tempo_words = ("quicker phrase", "slower phrase", "slower closing phrase",
                   "mid-tempo", "quickly", "steady closing tempo")
    action_units = units[1:] if units and any(w in units[0] for w in tempo_words) else units
    assert len(action_units) <= 3, ("too many action units for 5.06s: %d" % len(action_units), motion)
    # V103 关键 12（首段必须是腿部动作）—— 每轮的第一个动作单元驱动下半身，
    #   这正是「下盘带动、上身跟随」的时值结构，也是与 V92 最核心的差别。
    _first = action_units[0].lower() if action_units else ""
    assert any(w in _first for w in LEG_WORDS), ("first unit is not a leg action", motion)

    # V106: 参考图节点固定绑定 R01 初始底图（不随轮次更换）
    assert wf["13"]["inputs"]["image"] == ORIGINAL_BASE
    # V107: 已移除 prev_locked_tail_png 死代码分支 —— continue.json 只加载 .latent，
    #       tail_locked.png 从未被任何工作流节点引用。
    if use_bootstrap:
        assert wf["100"]["inputs"].get("use_prev_samples") is False
        assert "180" not in wf
        assert wf["100"]["inputs"]["motion_latent_count"] == MLC_BOOTSTRAP
    else:
        assert wf["100"]["inputs"].get("use_prev_samples") is True
        assert wf["100"]["inputs"]["motion_latent_count"] == MLC_CONTINUE
        assert wf["100"]["inputs"]["prev_samples"] == ["180", 0]
        assert wf["180"]["inputs"]["latent"] == LATENT_INPUT_NAME

    return wf, seed


# HTTP direct loopback opener
def _comfy_opener():
    """显式绕过本地/系统 HTTP 代理，确保回环 ComfyUI 直连。"""
    import urllib.request
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def queue_prompt(wf):
    import urllib.request
    data = json.dumps({"prompt": wf}).encode("utf-8")
    req = urllib.request.Request(API_URL, data=data, headers={"Content-Type": "application/json"})
    with _comfy_opener().open(req, timeout=60) as resp:
        return json.loads(resp.read())


def wait_for_completion(prompt_id, timeout_min=ROUND_TIMEOUT_MIN):
    t0 = time.time()
    opener = _comfy_opener()
    while True:
        if time.time() - t0 > timeout_min * 60:
            raise TimeoutError(f"round exceeded {timeout_min} min (prompt_id={prompt_id})")
        try:
            with opener.open(HISTORY_URL + "/" + prompt_id, timeout=30) as resp:
                hist = json.loads(resp.read())
            if prompt_id in hist:
                entry = hist[prompt_id]
                status = (entry.get("status") or {})
                if status.get("completed") or status.get("status_str") in (
                    "success", "error", "interrupted"
                ):
                    if status.get("status_str") == "error":
                        raise RuntimeError(f"ComfyUI error: {status.get('messages')}")
                    # ComfyUI 命中缓存时 outputs 为空 {}，不会写新的 mp4。
                    # 必须当失败处理（并换种子），否则会死循环。
                    if not (entry.get("outputs") or {}):
                        raise RuntimeError("ComfyUI returned cached prompt with empty outputs "
                                           "(cache hit -> no new file written)")
                    return entry
        except RuntimeError:
            raise
        except Exception:
            pass
        time.sleep(5)


def find_mp4(round_idx, before=None):
    files = sorted(
        glob.glob(os.path.join(COMFYUI_OUTPUT, f"{OUTPUT_PREFIX}_R{round_idx+1:02d}*.mp4")),
        key=os.path.getmtime,
    )
    if before is not None:
        fresh = [f for f in files if f not in before]
        return fresh[-1] if fresh else None
    return files[-1] if files else None


def find_latent(round_idx, prefix=None):
    prefix = prefix or OUTPUT_PREFIX
    files = sorted(
        glob.glob(
            os.path.join(COMFYUI_OUTPUT, "**", f"{prefix}_R{round_idx+1:02d}*FINAL*.latent"),
            recursive=True,
        ),
        key=os.path.getmtime,
    )
    return files[-1] if files else None


def install_latent(round_idx, prefix=None):
    src = find_latent(round_idx, prefix)
    if not src:
        raise RuntimeError(f"no latent for R{round_idx+1:02d} under {COMFYUI_OUTPUT}")
    dst = os.path.join(COMFYUI_INPUT, LATENT_INPUT_NAME)
    shutil.copy2(src, dst)
    if not os.path.exists(dst) or os.path.getsize(dst) < 100:
        raise RuntimeError(f"latent copy failed or empty: {dst}")
    print(f"[LATENT] {src} -> {dst} ({os.path.getsize(dst)} bytes)")
    return True


def copy_mp4(round_idx, mp4_path):
    rd = os.path.join(OUTPUT_DIR, f"Round{round_idx+1:02d}")
    os.makedirs(rd, exist_ok=True)
    dst = os.path.join(rd, os.path.basename(mp4_path))
    shutil.copy2(mp4_path, dst)
    return dst


def check_pops(path, w=208, h=367):  # 544x960 同比例；V103 的 208x304 会压扁
    """返回 dict(ok, bad=[(step,frac%,ratio)], max_frac, max_ratio)；无 cv2 或探测失败返回 None"""
    if not HAVE_CV:
        return None
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height", "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True)
        W, H = map(int, r.stdout.strip().split(",")[:2])
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "gray", "-"],
            capture_output=True, check=True).stdout
    except Exception as e:
        print(f"[QC WARN] 探测失败，跳过本轮质检: {e}")
        return None
    a = np.frombuffer(raw, np.uint8)
    n = a.size // (W * H)
    a = a[: n * W * H].reshape(n, H, W)
    g = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) for f in a]
    frac = []
    for i in range(len(g) - 1):
        d = np.abs(g[i + 1] - g[i])
        frac.append(float((cv2.GaussianBlur(d, (0, 0), 3) > 30).mean() * 100))
    frac = np.array(frac)
    bad, max_ratio = [], 0.0
    for i in range(len(frac)):
        neigh = np.r_[frac[max(0, i - 6):i], frac[i + 1:i + 7]]
        med = float(np.median(neigh)) if len(neigh) else 1.0
        q = float(frac[i]) / max(med, 1.0)
        max_ratio = max(max_ratio, q)
        if frac[i] >= POP_FRAC and q >= POP_RATIO:
            bad.append((i, float(frac[i]), q))
    return dict(ok=not bad, bad=bad, max_frac=float(frac.max()), max_ratio=max_ratio)


def log_qc(round_no, attempt, seed, qc):
    path = os.path.join(OUTPUT_DIR, "qc_log.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["round", "attempt", "seed", "ok", "max_changed_pct", "max_ratio", "bad_steps"])
        wr.writerow([round_no, attempt, seed, qc["ok"], round(qc["max_frac"], 2),
                     round(qc["max_ratio"], 2), ";".join(f"{s}:{fr:.1f}%x{q:.1f}" for s, fr, q in qc["bad"])])


def setup_resume(start_round, prev_dir, collected):
    """复制前 1..start-1 轮 mp4 到本输出目录，并装入第 start-1 轮的 latent"""
    for k in range(1, start_round):
        files = sorted(glob.glob(os.path.join(prev_dir, f"Round{k:02d}", "*.mp4")), key=os.path.getmtime)
        if not files:
            raise RuntimeError(f"resume: no mp4 for Round{k:02d} in {prev_dir}")
        rd = os.path.join(OUTPUT_DIR, f"Round{k:02d}")
        os.makedirs(rd, exist_ok=True)
        dst = os.path.join(rd, os.path.basename(files[-1]))
        if not os.path.exists(dst):
            shutil.copy2(files[-1], dst)
        collected.append(dst)
    print(f"[RESUME] reused R01..R{start_round-1:02d} mp4 from {prev_dir}")
    install_latent(start_round - 2, prefix=PREV_PREFIX)
    print(f"[RESUME] prev latent = {PREV_PREFIX} R{start_round-1:02d} FINAL")


def trim_head_ffmpeg(src, dst, drop_frames):
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-vf", f"select=gte(n\\,{drop_frames}),setpts=PTS-STARTPTS",
        "-an", "-r", str(FPS), "-c:v", "libx264", "-crf", "16",
        "-pix_fmt", "yuv420p", dst,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print("[TRIM FAIL]", r.stderr.decode(errors="ignore")[:400])
        shutil.copy2(src, dst)
        return dst, False
    return dst, True


def merge_with_drop(round_paths):
    trimmed_dir = os.path.join(OUTPUT_DIR, "_trimmed")
    os.makedirs(trimmed_dir, exist_ok=True)
    pieces = []
    for i, p in enumerate(round_paths):
        out_p = os.path.join(trimmed_dir, f"seg_{i+1:02d}.mp4")
        if i == 0:
            shutil.copy2(p, out_p)
            print(f"[TRIM] R01 keep all -> {out_p}")
        else:
            trim_head_ffmpeg(p, out_p, DROP_HEAD_FROM_R2)
            print(f"[TRIM] R{i+1:02d} drop first {DROP_HEAD_FROM_R2} frames -> {out_p}")
        pieces.append(out_p)

    list_file = os.path.join(OUTPUT_DIR, "concat_list.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for p in pieces:
            ap = os.path.abspath(p).replace("\\", "/")
            f.write(f"file '{ap}'\n")
    out = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_merged_trim{DROP_HEAD_FROM_R2}.mp4")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", out]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[MERGE OK] {out}")
        return out
    except FileNotFoundError:
        print("[MERGE] ffmpeg not in PATH; use concat_list.txt manually")
        return None
    except subprocess.CalledProcessError as e:
        print("[MERGE FAIL]", (e.stderr or b"").decode(errors="ignore")[:500])
        return None


def main():
    global COMFY_URL, API_URL, HISTORY_URL, COMFY_ROOT, COMFYUI_INPUT, COMFYUI_OUTPUT, PYTHON_CANDIDATES
    global ORIGINAL_BASE, DROP_HEAD_FROM_R2, MOTION, CFG, OUTPUT_PREFIX, OUTPUT_DIR, LATENT_INPUT_NAME, LATENT_SAVE_DIR, PREV_PREFIX, PREV_DIR_DEFAULT

    raw_args = sys.argv[1:]
    if "--comfy-root" in raw_args:
        COMFY_ROOT = raw_args[raw_args.index("--comfy-root") + 1]
        COMFYUI_INPUT = os.path.join(COMFY_ROOT, "ComfyUI", "input")
        COMFYUI_OUTPUT = os.path.join(COMFY_ROOT, "ComfyUI", "output")
        PYTHON_CANDIDATES = _build_python_candidates(COMFY_ROOT)
    if "--comfy-url" in raw_args:
        COMFY_URL = raw_args[raw_args.index("--comfy-url") + 1].rstrip("/")
        API_URL = f"{COMFY_URL}/prompt"
        HISTORY_URL = f"{COMFY_URL}/history"

    ensure_qc_python()

    args = sys.argv[1:]
    if "--output-dir" in args:
        OUTPUT_DIR = os.path.abspath(args[args.index("--output-dir") + 1])
        PREV_DIR_DEFAULT = OUTPUT_DIR
    if "--base-image" in args:
        ORIGINAL_BASE = args[args.index("--base-image") + 1]
    if "--motion" in args:
        MOTION = float(args[args.index("--motion") + 1])
    if "--cfg" in args:
        CFG = float(args[args.index("--cfg") + 1])
    if "--tag" in args:
        tag = args[args.index("--tag") + 1]
        OUTPUT_PREFIX = f"{OUTPUT_PREFIX}_{tag}"
        if "--output-dir" not in args:
            OUTPUT_DIR = f"{OUTPUT_DIR}_{tag}"
        LATENT_INPUT_NAME = LATENT_INPUT_NAME.replace(".latent", f"_{tag}.latent")
        LATENT_SAVE_DIR = f"{LATENT_SAVE_DIR}_{tag}"
        PREV_PREFIX = f"{PREV_PREFIX}_{tag}"
        PREV_DIR_DEFAULT = OUTPUT_DIR
    dry_run = "--dry-run" in args
    no_merge = "--no-merge" in args
    use_qc = ("--no-qc" not in args) and HAVE_CV
    rounds = int(args[args.index("--rounds") + 1]) if "--rounds" in args else (
        TEST_ROUNDS if TEST_MODE else TOTAL_ROUNDS)
    start = int(args[args.index("--start-round") + 1]) if "--start-round" in args else 1
    prev_dir = args[args.index("--prev-dir") + 1] if "--prev-dir" in args else PREV_DIR_DEFAULT
    if "--prev-prefix" in args:
        globals()["PREV_PREFIX"] = args[args.index("--prev-prefix") + 1]
    if "--drop" in args:
        DROP_HEAD_FROM_R2 = int(args[args.index("--drop") + 1])
    assert 1 <= start <= rounds

    print("=" * 76)
    print("[START] FUKE_ShortSegV109  (24 轮 x 5 秒全身动作 544x960; P1 内核 + LAB 轮间色彩锚定)")
    print(f"  CM={CM_FACTOR} LAB | lr={LATENT_REFRESH} | motion={MOTION} | cfg={CFG} | steps 2+4+4 | MLC={MLC_BOOTSTRAP}/{MLC_CONTINUE} | base={ORIGINAL_BASE}")
    print(f"  res={RESOLUTION_W}x{RESOLUTION_H} | fps={FPS} | frames={FRAMES_PER_ROUND} | duration={FRAMES_PER_ROUND/FPS:.2f}s")
    print(f"  rounds={start}..{rounds}  out={OUTPUT_DIR}")
    print(f"  QC={'ON' if use_qc else 'OFF'} (frac>={POP_FRAC}% & x{POP_RATIO}, retries={QC_RETRIES})  drop={DROP_HEAD_FROM_R2}")
    if not HAVE_CV and "--no-qc" not in args:
        print("  [WARN] cv2/numpy 不可用，自动质检已关闭")
    if start > 1:
        print(f"  resume from {PREV_PREFIX} R{start-1:02d}  prev-dir={prev_dir}")
    print("=" * 76)

    if dry_run:
        for ri in range(start - 1, rounds):
            wf, seed = build_round_workflow(ri)
            if ri < start - 1 + 4:
                prev = "none" if ri == 0 else wf["100"]["inputs"].get("prev_samples")
                print(f"[DRY] R{ri+1:02d} seed={seed} prev={prev} CM={wf['150']['inputs']['factor']} "
                      f"motion={wf['100']['inputs']['motion']} lr={wf['100']['inputs']['latent_refresh']} "
                      f"MLC={wf['100']['inputs']['motion_latent_count']}")
                print(f"       prompt={wf['50']['inputs']['text']}")
        print(f"[DRY-RUN OK] built+asserted {rounds - start + 1} rounds, nothing queued")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ensure_base_image_in_input()

    collected = []
    if start == 1:
        prev_latent_path = os.path.join(COMFYUI_INPUT, LATENT_INPUT_NAME)
        if os.path.exists(prev_latent_path):
            os.remove(prev_latent_path)
            print(f"[CLEAN] removed stale prev latent: {prev_latent_path}")
    else:
        setup_resume(start, prev_dir, collected)

    flagged = []
    for ri in range(start - 1, rounds):
        print(f"\n======== ROUND {ri+1}/{rounds} ========")
        ok = False
        qc_tries, seed_offset = 0, 0
        pre_existing = set(glob.glob(os.path.join(COMFYUI_OUTPUT, f"{OUTPUT_PREFIX}_R{ri+1:02d}*.mp4")))
        for attempt in range(MAX_RETRIES + QC_RETRIES + 1):
            try:
                wf, seed = build_round_workflow(ri, seed_offset)
                print(f"[SEED] {seed}")
                print(f"[CFG] prev={wf['100']['inputs'].get('prev_samples')} "
                      f"CM={wf['150']['inputs']['factor']} lr={wf['100']['inputs']['latent_refresh']}")
                print(f"[PROMPT] {wf['50']['inputs']['text'][:160]}...")
                res = queue_prompt(wf)
                pid = res.get("prompt_id")
                print(f"[QUEUE] prompt_id={pid} attempt={attempt+1}")
                _t0 = time.time()
                hist = wait_for_completion(pid)
                _dt = time.time() - _t0
                if _dt < MIN_ROUND_SECONDS:
                    raise RuntimeError(f"round finished in {_dt:.1f}s (<{MIN_ROUND_SECONDS}s)，疑似缓存命中/假成功")
                status = (hist.get("status") or {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI error: {status.get('messages')}")
                time.sleep(2)
                mp4 = find_mp4(ri, before=pre_existing)
                if not mp4:
                    raise RuntimeError(f"no mp4 for R{ri+1:02d}")

                if use_qc:
                    qc = check_pops(mp4)
                    if qc is not None:
                        log_qc(ri + 1, attempt + 1, seed, qc)
                        print(f"[QC] max_changed={qc['max_frac']:.1f}% max_ratio={qc['max_ratio']:.1f} "
                              f"{'PASS' if qc['ok'] else 'POP ' + str(qc['bad'])}")
                        if not qc["ok"]:
                            if qc_tries < QC_RETRIES:
                                qc_tries += 1
                                seed_offset += SEED_RETRY_STEP
                                pre_existing.add(mp4)
                                print(f"[QC] retry {qc_tries}/{QC_RETRIES} with new seed (+{seed_offset})")
                                continue
                            print("[QC WARN] retries exhausted, keeping last attempt")
                            flagged.append(ri + 1)

                dst = copy_mp4(ri, mp4)
                print(f"[OK] {dst}")
                # V107: 已移除 color_lock_tail_frame 死代码块 —— continue.json 只加载
                #       .latent，tail_locked.png 从未被任何工作流节点引用。
                collected.append(dst)
                # 无论是否最后一轮都装入 latent：否则 UI 里手动跑 continue 工作流时
                # LoadLatent 找不到 prev.latent，校验失败 ~0.7s 就"过"了
                install_latent(ri)
                print(f"[HANDOFF OK] R{ri+1:02d} -> R{ri+2:02d} (latent 已就位)")
                ok = True
                break
            except Exception as e:
                print(f"[ERROR] R{ri+1:02d} attempt {attempt+1}: {e}")
                # 失败必须换种子，否则 ComfyUI 缓存命中 -> 空 outputs -> 死循环
                seed_offset += SEED_RETRY_STEP
                time.sleep(3)
        if not ok:
            print(f"[FAIL] Round {ri+1} after retries. Collected={len(collected)}")
            break

    if not no_merge and collected:
        print(f"\n======== MERGE (R2+ drop {DROP_HEAD_FROM_R2} frames) ========")
        merged = merge_with_drop(collected)
        if merged:
            print(f"[DONE] merged: {merged}")
    if flagged:
        print(f"\n[QC SUMMARY] 重试后仍命中传送检测的轮: {flagged}，请肉眼复核（qc_log.csv）")
    print("\n[DONE] V109（544x960 + P1 黄金内核 + LAB 轮间色彩锚定）。每轮 81 帧 @16FPS；")
    print(f"      后处理建议: python postprocess_24R_V109.py --stage all --root \"{OUTPUT_DIR}\"")


if __name__ == "__main__":
    main()
