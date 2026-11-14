# -*- coding: utf-8 -*-
"""
FUKE_ShortSegV92_Claude - 有结构、有表现力的长视频编排（Claude 版）。基于 G3g，只改提示词

依据:
  - 你与 GPT 的成片评价: 色彩/身份/服装/背景较好(保持不动)；动作幅度偏小、姿势重复、缺少段落变化；
    构图是头到大腿，下半身基本不在画面里。
  - 我的反思: G3g 的重音轮写的是踢腿/跺脚/踏步，这些在头-大腿构图里看不到，力度被浪费。
    V92 的编舞只用画面里看得见的部位: 手臂、肩、胸、躯干、髋、头、视线、手(脸/头发/颈)。
  - 艺术表现力靠"结构"而不是平均速度: 24 轮 x 2.7 秒 = 约 65 秒，按一首歌的段落编排。

V92 的编排（只改提示词与词表检查，参数/工作流/颜色/MLC/QC 与 G3g 完全相同）:
  段落   轮次      作用
  序     R01-R03   克制、自信，亮出"手势母题 A"
  主歌   R04-R08   躯干/髋/肩的律动，R06 呼吸
  预副歌 R09-R11   张力上升，R11 沉下去（副歌前的蓄力）
  副歌 1 R12-R15   重音峰 R12、R15，母题 A 放大、母题 B 出现
  间奏   R16-R17   降低重心、侧脸、视线变化
  副歌 2 R18-R21   重音峰 R18、R21，R19 慢速蓄力
  尾声   R22-R24   R23 终章重音（母题 A 双手版），R24 收束
  - 母题 A: "her right hand draws a line from her hip up to her cheek and pushes out toward the camera"
    在 R01 / R12 / R23 回归（每次不同的搭配动作）
  - 母题 B: "her arms cross in front of her chest and snap open wide" 在 R05 / R15 / R21 回归
  - 每轮聚焦不同身体部位 + 不同朝向，避免"重复同一手势"
  - 表情/视线随段落变化: 冷静 -> 紧绷 -> 凌厉 -> 低垂 -> 收束；仍禁止娇柔词
  - 类别: M 中速 8 / Q 偏快 6 / P 重音峰 5 / S 慢速 5；powerful 只出现在 P 轮；P 轮间隔 >= 2 轮

G3g 24R 实测教训（后处理报告，全片 flow_mean 5.78，S/M/F 档 21/24 达标）:
  - 速度词没有兑现: 我标成 S(慢)的 R04 是 6.2、R20 是 6.3；标成 P(重音)的 R02 只有 3.7、R07 只有 4.6。
    真正拉高光流的是动作内容: "on each / every beat" 的重复 + 交替 + 行进
    (R14 9.0、R21 8.1、R23 7.7、R22 7.0)；单次大动作 + "lands in a stance" 的轮偏低。
  - R01 是 bootstrap 轮(无 prev latent)，median 2.26、pause 0.25，是全片最弱；需要在 R01 里就写节拍重复。
  - 背景 bg/person 全片 0.021-0.154，说明 SCENE_LOCK 即使在 R14/R15(median 9)也锁得住。
  V92 据此修订: P 轮不再写单次动作，改成"三次重音 / 每拍 / 两次起落"；S 轮保持单一的持续动作，
  不写 on each beat；R01 写入"肩部两次击拍"。速度词仍保留，但不再指望它单独决定快慢。

不确定之处（需要你验证）:
  - 我没有看到 G3g 的逐轮光流数据，V92 的目标区间沿用 G3f/G3g 的校准(P 轮 6-7.5，其余 4-5.5)，不一定准确。
  - 构图限制: 头-大腿构图下无法表现腿部舞步。要真正全身舞蹈，需要换一张全身的 ORIGINAL 底图，
    本脚本加了 --base-image 开关（ComfyUI input 里的文件名），但换图会改变脸的像素占比，色彩/身份基准要重新验证。
  - 文字能否稳定传递"段落感"没有证据；先看 R01-R06，再决定是否跑满 24 轮。

用法:
  python run_fuke_shortsegV92_Claude.py --dry-run
  python run_fuke_shortsegV92_Claude.py --rounds 6        # 先跑 6 轮看开头
  python run_fuke_shortsegV92_Claude.py                    # 默认 24 轮（含自动质检重跑）
  其他: --tag S  --motion F  --cfg F  --base-image NAME  --start-round K  --drop N  --no-qc  --no-merge
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

API_URL = "http://127.0.0.1:8188/prompt"
HISTORY_URL = "http://127.0.0.1:8188/history"
COMFY_ROOT = r"E:\ComfyUI\ComfyUI-aki-v3"
COMFYUI_INPUT = os.path.join(COMFY_ROOT, "ComfyUI", "input")
COMFYUI_OUTPUT = os.path.join(COMFY_ROOT, "ComfyUI", "output")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 候选解释器：自动切到带 cv2/numpy 的 Python（否则自动质检会被静默关闭）
PYTHON_CANDIDATES = [
    os.path.join(COMFY_ROOT, "python", "python.exe"),
    os.path.join(COMFY_ROOT, "python312", "python.exe"),
    r"C:\Python315\python.exe",
]
REEXEC_GUARD = "FUKE_SHORTSEG_V92_REEXEC"

BOOTSTRAP_WORKFLOW_FILE = os.path.join(SCRIPT_DIR, "workflow_fuke_shortsegV92_Claude_bootstrap.json")
CONTINUE_WORKFLOW_FILE = os.path.join(SCRIPT_DIR, "workflow_fuke_shortsegV92_Claude_continue.json")

OUTPUT_DIR = os.path.join(
    SCRIPT_DIR, "output_fuke_shortsegV92_Claude_P1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_MODE = False
TEST_ROUNDS = 6
TOTAL_ROUNDS = 24
OUTPUT_PREFIX = "FUKE_ShortSegV92_Claude_P1"
LATENT_SAVE_DIR = "fuke_shortsegV92_Claude_P1_latent"
ORIGINAL_BASE = "grok-image-11990385-5fb2-4bca-81c4-ab097b339968.jpg"
SEED_BASE = 123456789

CM_FACTOR = 0.90
LATENT_REFRESH = 0.80
MOTION = 1.18
CFG = 3.0
DROP_HEAD_FROM_R2 = 4
FPS = 30
MLC_BOOTSTRAP = 1
MLC_CONTINUE = 1

LATENT_INPUT_NAME = "fuke_shortsegV92_Claude_P1_prev.latent"
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


# ---------------------------------------------------------------- 提示词层
OUTFIT_LOCK = (
    "same character identity as reference image, long straight glossy white hair, "
    "red eyes, cyan blue silk qipao cheongsam with gold floral embroidery, "
    "red frog buttons and red tassels, high mandarin collar, high side slit, "
    "no costume change, no hairstyle change, no extra hair accessories, no twin tails"
)

# SCENE_LOCK：与 G3e 逐字符相同（背景本身没问题，不做改动）
SCENE_LOCK = (
    "static camera, locked-off shot, the background and cherry blossom trees stay perfectly still, "
    "only a few pink petals drift down"
)

# ---------------------------------------------------------------- G3f 提示词层
#
# 本地现场观察（本版设计依据）：
#   G3  : R1-R3 正常；问题在"画面抖动 / 背景抖动"，出现在动作过快的几轮
#   G3b : 动作都慢，像摆 pose
#   G3d : 动作都慢，像摆 pose
#   G3e : 动作都慢，像摆 pose
#   背景本身没问题（G3e bg_flow 0.4-0.9 px/帧）。
#
# 所以 G3f = 正常人类舞蹈节奏：不快不慢、有快有慢，且不能出现 G3 那种"快到抖"。
#
# 节奏设计依据（人类可完成度）：
#   每轮 81 帧 @30fps = 2.70 秒。24 轮 = 64.8 秒 = 6 个乐句（每 4 轮约 10.8 秒），
#   正好落在"人类 2-10 秒能完成"的动作窗口内。
#   专业舞者在 2.7 秒内可完成约 4-8 个动作单元（2-3 个乐句），
#   所以每一轮写 2-3 个连贯动作，而不是 G3 那种"一轮一个超大爆发动作"（那是抖动的来源）。
#
# 与 G3e 的关系：
#   SCENE_LOCK / NEGATIVE(基础) / 全部生成参数与 G3e 完全一致，
#   唯一变量 = 运动提示词（动作幅度 + 动作数量 + 快慢交替）。指标差异可干净归因。

# 动作底座：不规定快慢，只规定"连贯、有表演感、不断拍、不定住"
DANCE_BASE = (
    "anime girl performing an expressive choreography with strong confident presence, "
    "cool confident expression, alternating quicker sharp accents with slower held shapes, "
    "upper body and hips leading, hits on the beat"
)

# 24 轮的类别: M 中速 / Q 偏快 / P 重音峰 / S 慢速呼吸
# 段落: 序 R01-03 | 主歌 R04-08 | 预副歌 R09-11 | 副歌1 R12-15 | 间奏 R16-17 | 副歌2 R18-21 | 尾声 R22-24
ROUND_CLASS = "MMQMQSMQMQSPQMPSMPSQPMPS"

# 手势母题（原文回归，搭配不同的动作）
MOTIF_A = "her right hand draws a line from her hip up to her cheek and pushes out toward the camera"
MOTIF_B = "her arms cross in front of her chest and snap open wide"
MOTIF_A_ROUNDS = (0, 11, 22)    # R01 / R12 / R23
MOTIF_B_ROUNDS = (4, 14, 20)    # R05 / R15 / R21

VARIATIONS = [
    "in an even mid-tempo, her right hand draws a line from her hip up to her cheek and pushes out toward the camera, her shoulders hitting the beat twice as her left arm extends wide, a cool confident gaze held on the camera",
    "in an even mid-tempo, her shoulders hit the beat alternately left and right, her torso twisting to follow, then her head tilts with a cool sideways glance",
    "on a quicker phrase, her torso rolls down and up twice, her hips hitting each beat, her arms opening wide on the second roll",
    "in an even mid-tempo, she steps to the right with her hips leading, her left arm sweeping across her body and her right hand rising to her hair, her gaze following the hand",
    "on a quicker phrase, her arms cross in front of her chest and snap open wide, then her shoulders pop twice on the beat",
    "on a slower phrase, she leans back from the waist with one arm extending long to the side, her chin lifting, then draws the arm back across her chest",
    "in an even mid-tempo, her hips draw a figure eight to the beat, both hands sliding from her waist to her ribs, her head turning to look over her shoulder",
    "on a quicker phrase, she turns a quarter to the right with her arms leading, snaps back to the front with a shoulder hit, her eyes locked on the camera",
    "in an even mid-tempo, her weight shifts left and right, her hands framing her face and then pushing outward, her eyes narrowing with intensity",
    "on a quicker phrase, both arms punch out to the sides in alternation, her chest popping with each punch, her head snapping toward the camera",
    "on a slower phrase, she lowers her body with a deep knee bend and her arms sinking low, holds the low shape for a beat, then rises with her gaze lifting to the camera",
    "on a quicker phrase with powerful sharp accents, her right hand draws a line from her hip up to her cheek and pushes out toward the camera, then three sharp accents with both arms thrusting out wide on each accent, a fierce gaze at the camera",
    "on a quicker phrase, she travels two wide steps to the left with her arms crossing and opening on each beat, her shoulders hitting the accents",
    "in an even mid-tempo, she turns a half turn to the left with her arms trailing wide, then faces the camera again with one hand at her hip and a confident smirk",
    "on a quicker phrase with powerful sharp accents, her arms cross in front of her chest and snap open wide, then her shoulders pivot sharply right and left on every beat, her arms driving out to the side each time",
    "on a slower phrase, she lowers into a low stance and sweeps one arm in a long arc along her side, her head turned in profile, her eyes half closed",
    "in an even mid-tempo, she rises and shifts her weight to the left, her hands sliding along her waist and her torso twisting, then her gaze turns back to the camera",
    "on a quicker phrase with powerful sharp accents, she travels three wide steps to the right with a sharp shoulder hit on each step, one arm slicing across her body each time, then lands in a strong open stance with both arms thrown wide",
    "on a slower phrase, she stands in a strong stance with her head tilted back, one hand gliding up from her waist along her neck, her eyes closing",
    "on a quicker phrase, she snaps her head toward the camera, her arms alternately thrusting up and out to the sides, her hips hitting each beat",
    "on a quicker phrase with powerful sharp accents, her arms cross in front of her chest and snap open wide, then she drops into a deep knee bend and rises twice on the beat, her arms driving down and out each time with a fierce gaze",
    "in an even mid-tempo, she rises with her arms opening wide on the rise, one hand sweeping across her chest, her torso twisting to the left",
    "on a quicker phrase with powerful sharp accents, both hands draw a line from her hips up to her cheeks and push out toward the camera, then she throws her arms wide with a sharp shoulder hit on every beat, her chin lifted and her eyes locked on the camera",
    "on a slower phrase, the dance resolves as she steps back and opens her arms wide toward the camera, her chin lifted, holding a confident final shape",
]

TEMPO_WORD = {"M": "mid-tempo", "Q": "quicker", "P": "quicker", "S": "slower"}

# 全部轮禁用: G3 抖动词(超大爆发)、G3b/d/e 摆 pose 词(慢速/定格)、
# 以及头-大腿构图里看不到的腿部动作词（写了也只是浪费提示词权重）
FORBIDDEN_ALWAYS = (
    "leap", "jump", "spin", "twirl", "whip", "fling", "explosive", "sprint", "run",
    "huge", "giant", "massive", "violent", "shake", "rapid", "bounce",
    "across the frame", "full-body", "dynamic freeze",
    "slowly", "gracefully", "gently", "delicately", "languid", "sluggish",
    "lazy", "dawdling", "freeze", "frozen", "posing", "stiff", "sliding into",
    "hold a pose", "in a pose", "natural", "smooth",
    "kick", "stomp", "heel", "toe", "footwork", "foot",
)
# 幅度限制词 / 娇柔词: 除 S 轮外禁用
FORBIDDEN_NON_SLOW = (
    "small", "short", "light", "in place", "calm", "settle", "rest", "elbows held close",
    "cheerful", "playful", "cute", "bright smile", "coy", "shy",
)

MOTION_PROMPTS = [f"{DANCE_BASE}, {v}" for v in VARIATIONS]

# NEGATIVE = G3e 原版 + G3f 追加的"反摆 pose"与"反抖动"项
NEGATIVE = (
    "blurry, low quality, distorted face, extra limbs, watermark, text, "
    "letters, words, signature, logo, subtitles, caption, typography, "
    "written characters, Chinese characters, English text, glyphs, "
    "pale washed out colors, desaturated, grayish, "
    "different outfit, costume change, clothing change, hair color change, "
    "twin tails, ponytail, short hair, different dress, kimono overlay, "
    "extra hair ornaments changing every frame, "
    "camera shake, camera movement, handheld, zoom in, zoom out, "
    "shaking background, swaying trees, trees moving, wind blowing branches, "
    "background motion, jitter, flicker, motion blur, "
    # G3f 追加：反摆 pose
    "slow motion, sluggish movement, lazy motion, frozen pose, stiff pose, "
    "holding a pose, posing for the camera, standing still, motionless, "
    # G3f 追加：反抖动
    "violent shaking, frame jitter, screen shake, excessive motion blur, "
    "whip pan, violent hair whip, distorted motion, "
    # G3g 追加：反娇柔
    "cutesy, coy pose, shy timid posture, bashful"
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


def build_round_workflow(round_idx, seed_offset=0):
    use_bootstrap = (round_idx == 0)
    path = BOOTSTRAP_WORKFLOW_FILE if use_bootstrap else CONTINUE_WORKFLOW_FILE
    wf = load_workflow(path)

    motion = MOTION_PROMPTS[round_idx % len(MOTION_PROMPTS)]
    prompt = f"{motion}, {SCENE_LOCK}, {OUTFIT_LOCK}"
    wf["50"]["inputs"]["text"] = prompt
    wf["51"]["inputs"]["text"] = NEGATIVE

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

    wf["100"]["inputs"]["latent_refresh"] = LATENT_REFRESH
    wf["100"]["inputs"]["motion"] = MOTION
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
    assert abs(float(wf["100"]["inputs"]["latent_refresh"]) - LATENT_REFRESH) < 1e-6
    assert abs(float(wf["100"]["inputs"]["motion"]) - MOTION) < 1e-6
    assert wf["160"]["inputs"]["images"] == ["150", 0]
    # 生成参数与 G3e 一致
    assert wf["32"]["inputs"]["value"] == 10                      # 步数 2+4+4
    assert (wf["101"]["inputs"]["end_at_step"], wf["102"]["inputs"]["end_at_step"]) == (2, 6)
    assert wf["13"]["inputs"]["image"] == ORIGINAL_BASE
    # 文案：动作在前、场景锁、造型锁、负面词
    assert prompt.startswith(motion)
    assert SCENE_LOCK in prompt
    assert OUTFIT_LOCK in prompt
    assert "qipao" in prompt or "cheongsam" in prompt
    assert motion.startswith(DANCE_BASE)
    # G3g: 按轮次类别检查词表
    cls = ROUND_CLASS[round_idx % len(ROUND_CLASS)]
    var = motion.replace(DANCE_BASE, "")
    assert not any(w in var for w in FORBIDDEN_ALWAYS), var
    if cls != "S":
        assert not any(w in var for w in FORBIDDEN_NON_SLOW), var
    assert ("powerful" in var) == (cls == "P"), var
    assert TEMPO_WORD[cls] in var, var
    # V92: 母题必须在指定轮次原文出现，且不出现在别处
    ri24 = round_idx % len(ROUND_CLASS)
    assert (("a line from her hip" in var) == (ri24 in MOTIF_A_ROUNDS)), var
    assert ((MOTIF_B in var) == (ri24 in MOTIF_B_ROUNDS)), var
    assert cls != "P" or ri24 in (11, 14, 17, 20, 22)
    # G3f 关键 2：SCENE_LOCK / NEGATIVE 与 G3e 基线一致（背景不动）
    assert "stay perfectly still" in SCENE_LOCK
    assert "a few pink petals drift down" in SCENE_LOCK
    assert "camera shake" in NEGATIVE
    assert "letters" in NEGATIVE and "signature" in NEGATIVE
    # G3f 关键 3：负面词必须同时含"反摆 pose"与"反抖动"
    assert "holding a pose" in NEGATIVE and "standing still" in NEGATIVE
    assert "coy pose" in NEGATIVE
    assert "frame jitter" in NEGATIVE and "violent shaking" in NEGATIVE
    # G3f 关键 4：快慢必须成对出现（既有 quicker 也有 slower），不能全片一种速度
    assert ("quicker" in motion) or ("slower" in motion) or ("mid-tempo" in motion)  # DANCE_BASE 含 quicker/slower
    # G3f 关键 5：变长不能让每轮动作数超过人类 2.7 秒可完成的上限
    #   计数 = 逗号/连词分段的近似动作单元数
    units = [u for u in motion.replace(DANCE_BASE, "").replace(SCENE_LOCK, "")
             .replace(OUTFIT_LOCK, "").split(",") if u.strip()]
    assert len(units) <= 4, ("too many action units for 2.7s: %d" % len(units), motion)

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


def queue_prompt(wf):
    import urllib.request
    data = json.dumps({"prompt": wf}).encode("utf-8")
    req = urllib.request.Request(API_URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def wait_for_completion(prompt_id, timeout_min=ROUND_TIMEOUT_MIN):
    import urllib.request
    t0 = time.time()
    while True:
        if time.time() - t0 > timeout_min * 60:
            raise TimeoutError(f"round exceeded {timeout_min} min (prompt_id={prompt_id})")
        try:
            with urllib.request.urlopen(HISTORY_URL + "/" + prompt_id, timeout=30) as resp:
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


def check_pops(path, w=208, h=304):
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
    global ORIGINAL_BASE, DROP_HEAD_FROM_R2, MOTION, CFG, OUTPUT_PREFIX, OUTPUT_DIR, LATENT_INPUT_NAME, LATENT_SAVE_DIR, PREV_PREFIX, PREV_DIR_DEFAULT
    ensure_qc_python()

    args = sys.argv[1:]
    if "--base-image" in args:
        ORIGINAL_BASE = args[args.index("--base-image") + 1]
    if "--motion" in args:
        MOTION = float(args[args.index("--motion") + 1])
    if "--cfg" in args:
        CFG = float(args[args.index("--cfg") + 1])
    if "--tag" in args:
        tag = args[args.index("--tag") + 1]
        OUTPUT_PREFIX = f"{OUTPUT_PREFIX}_{tag}"
        OUTPUT_DIR = f"{OUTPUT_DIR}_{tag}"
        LATENT_INPUT_NAME = LATENT_INPUT_NAME.replace(".latent", f"_{tag}.latent")
        LATENT_SAVE_DIR = f"{LATENT_SAVE_DIR}_{tag}"
        PREV_PREFIX = f"{PREV_PREFIX}_{tag}"
        PREV_DIR_DEFAULT = OUTPUT_DIR
        os.makedirs(OUTPUT_DIR, exist_ok=True)
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
    print("[START] FUKE_ShortSegV92_Claude  (G3g + 段落编排/手势母题/只用画面内部位; flags --motion/--cfg/--tag/--base-image)")
    print(f"  CM={CM_FACTOR} LAB | lr={LATENT_REFRESH} | motion={MOTION} | cfg={CFG} | steps 2+4+4 | MLC=1 | base={ORIGINAL_BASE}")
    print(f"  rounds={start}..{rounds}  out={OUTPUT_DIR}")
    print(f"  QC={'ON' if use_qc else 'OFF'} (frac>={POP_FRAC}% & x{POP_RATIO}, retries={QC_RETRIES})  drop={DROP_HEAD_FROM_R2}")
    if not HAVE_CV and "--no-qc" not in args:
        print("  [WARN] cv2/numpy 不可用，自动质检已关闭")
    if start > 1:
        print(f"  resume from {PREV_PREFIX} R{start-1:02d}  prev-dir={prev_dir}")
    print("=" * 76)

    if dry_run:
        for ri in range(start - 1, min(rounds, start - 1 + 4)):
            wf, seed = build_round_workflow(ri)
            prev = "none" if ri == 0 else wf["100"]["inputs"].get("prev_samples")
            print(f"[DRY] R{ri+1:02d} seed={seed} prev={prev} CM={wf['150']['inputs']['factor']} "
                  f"lr={wf['100']['inputs']['latent_refresh']} MLC={wf['100']['inputs']['motion_latent_count']}")
            print(f"       prompt={wf['50']['inputs']['text']}")
        print("[DRY-RUN OK] built+asserted, nothing queued")
        return

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
                hist = wait_for_completion(pid)
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
                collected.append(dst)
                if ri < rounds - 1:
                    install_latent(ri)
                    print(f"[HANDOFF OK] R{ri+1:02d} -> R{ri+2:02d}")
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
    print("\n[DONE] V92-Claude. 接缝处理: fix_seams_retime.py --root <本输出目录> --drop 4|5")
    print("      指标对比: workspace\\scripts\\round_flow_metrics.py (cfg 里填本目录 Round01..Round18)")


if __name__ == "__main__":
    main()
