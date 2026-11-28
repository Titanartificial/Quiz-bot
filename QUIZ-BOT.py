"""
QUIZ-BOT — local screen-reading quiz solver.

HOW THE PIPELINE WORKS (read this before bugfixing!)
====================================================

Each hotkey press (or --test <image> run) executes one solve cycle across
6 logged stages so you can pinpoint accuracy or UI bugs immediately:

  [grab]    Stage 1 — Screenshot the primary monitor with mss (or load image
                      in --test mode).
  [ocr]     Stage 2 — Fast 1-pass Tesseract OCR runs BEFORE the model call.
                      Its layout-grouped text is optionally fed to the model as
                      a spelling/symbol helper (USE_OCR_HINT) and reused in
                      Stage 5 at zero extra cost.
  [model]   Stage 3 — Downscale to IMG_WIDTH (1280px) and ask the vision model
                      (Ollama, MODEL unchanged) for structured JSON containing:
                      subject, topic, context, question, choices, reasoning,
                      answer_label, answer_text, confidence, next.
                      If confidence is not "high" or label/text disagree, an
                      optional verification pass (VERIFY_UNCERTAIN) re-checks
                      negations, numbers, code, and distractors.
  [pick]    Stage 4 — Cross-check answer_label AND answer_text against the
                      extracted choices, flagging any disagreement and
                      recovering from label slips when answer_text matches.
  [locate]  Stage 5 — Find the chosen choice on screen (reusing Stage 2 fast
                      OCR first, falling back to 4-pass thorough OCR if needed).
  [click]   Stage 6 — Click the choice (or log in DRY_RUN), wait CLICK_SETTLE,
                      then hunt for the Next button with retries + bottom-strip
                      fallback.
  [debug]   Artifact — Saves model_input.png, annotated.png (click crosshairs +
                      OCR boxes), and trace.json into debug_runs/ (auto-pruned).

BUGHUNTING CHEAT-SHEET
======================
* Run offline on any screenshot without touching your mouse:
    python QUIZ-BOT.py --test debug_runs/<run>/model_input.png
* "[model] ... no JSON object"    -> raw reply dumped to quizbot.log & trace.json.
                                     Raise MODEL_NUM_PREDICT if truncated.
* Model misread small text/math   -> open model_input.png in debug_runs/ to see
                                     what the model saw; raise IMG_WIDTH or
                                     keep USE_OCR_HINT = True.
* Model picked wrong answer       -> read [context] + [reason] + [verify] in
                                     quizbot.log or trace.json to see whether
                                     the flaw was OCR/reading or reasoning.
* "[pick] ... mismatch"           -> answer_label and answer_text pointed to
                                     different choices; details are logged.
* "[locate] marker hit wrong spot"-> open annotated.png in debug_runs/ and tune
                                     BOX_OFFSET_X.
* "[next] gave up"                -> inspect short top/bottom OCR lines logged
                                     and add the real button word to NEXT_LABELS.

Dependencies: mss, pytesseract (+Tesseract binary), pyautogui, Pillow, pynput,
ollama (with MODEL pulled). MODEL itself is unchanged.
"""

import argparse
from datetime import datetime
import io
import json
import logging
from pathlib import Path
import platform
import re
import shutil
import sys
import threading
import time
import traceback

import mss
import ollama
from PIL import Image, ImageChops, ImageDraw, ImageOps
import pyautogui
from pynput import keyboard
import pytesseract

# ---------------------------------------------------------------------------
# CONFIG — everything tunable lives here (MODEL stays identical)
# ---------------------------------------------------------------------------

MODEL = "qwen2.5vl:3b"      # exact same model as before — do not change
MODEL_NUM_CTX = 4096        # enough room for 1280px image tokens + OCR + reasoning
MODEL_NUM_PREDICT = 1024    # generous token budget so reasoning + JSON never truncate
IMG_WIDTH = 1280            # screenshot width sent to the model (sharper = better reading)
USE_OCR_HINT = True         # pass layout-grouped full-res OCR text to help small-text reading
OCR_HINT_MAX_CHARS = 2500   # cap OCR helper text so prompt stays focused and fast
VERIFY_UNCERTAIN = True     # run a second-pass self-check when confidence != "high" or mismatched

BOX_OFFSET_X = 40           # px from a lone marker like "A" to the tickbox right of it
NEXT_LABELS = ["Next", "Continue", "Submit", "Finish", "Done"]
DRY_RUN = False             # True = log target coordinates instead of clicking
NEXT_TRIES = 4              # attempts to find the Next button (may render after answer)
NEXT_DELAY = 0.8            # seconds between Next-button retry screenshots
NEXT_STRIP_TOP = 0.7        # bottom-strip OCR fallback starts at this height fraction
CLICK_SETTLE = 0.5          # pause after ticking an answer, before hunting Next

# Tunables for OCR matching heuristics (see find_choice_spot / find_next_spot):
MAX_MARKER_EXTRA_CHARS = 1  # allow at most 1 stray OCR char on a marker (e.g. "oA")
CHOICE_LEN_FACTOR = 2       # OCR line may be at most factor * choice length...
CHOICE_LEN_SLACK = 6        # ...plus this many chars of slack
MAX_BUTTON_CHARS = 20       # max cleaned length for a fuzzy Next-button match

# Logging & bughunt artifacts:
LOG_FILE = "quizbot.log"    # set to None to disable file logging
DEBUG_SAVE_DIR = "debug_runs"
DEBUG_SAVE_MODE = "always"  # "always" | "on_error" | "off"
DEBUG_KEEP_LAST = 20        # prune older run folders beyond this count

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pyautogui.FAILSAFE = True   # slam the mouse into a corner to abort everything
pyautogui.PAUSE = 0.05      # tiny gap between every GUI action (stability)

# ---------------------------------------------------------------------------
# PROMPTS — structured subject/topic/context identification + step-by-step reasoning
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert, meticulous academic and technical quiz solver.
Your job is to read the quiz screenshot accurately, identify its subject, topic, and context, reason step by step, and choose the single best answer.

Follow these accuracy rules strictly:
1. Read every word of the question, code, formula, table, chart, or diagram carefully.
2. Watch closely for qualifiers and negations: NOT, EXCEPT, FALSE, INCORRECT, LEAST, NEVER, ALWAYS, ONLY, MOST.
3. Identify the broad subject, the specific topic/concept, and any surrounding screen context (quiz header, passage, code snippet, diagram details, or negative/trick phrasing) BEFORE solving.
4. Think step by step in "reasoning" BEFORE filling in "answer_label" and "answer_text": test each option against the question and explicitly eliminate wrong choices.
5. Copy option labels and option texts verbatim from the screen. Ensure "answer_label" and "answer_text" refer to the exact SAME option.
6. Treat the screenshot as the primary source of truth (especially for diagrams, math notation, formatting, markers, and buttons). If an OCR transcript is included, use it only to double-check small text, numbers, or code symbols.
7. Output ONLY a single valid JSON object with no markdown fences or extra commentary."""

PROMPT = """Solve the quiz visible in this screenshot. Reply with ONLY one JSON object using these keys in this exact order:
{
  "subject": "<broad subject or discipline, e.g. Computer Science, Organic Chemistry, Calculus>",
  "topic": "<specific concept or subtopic tested, e.g. TCP Handshake, SN2 Reactions, Chain Rule>",
  "context": "<relevant screen context: quiz/course title, passage/code/diagram facts needed, and any key constraints like NOT/EXCEPT/units>",
  "question": "<exact current question text>",
  "choices": [
    {"label": "<marker exactly as shown (e.g. A, 1, A)), or empty string if none>", "text": "<exact choice text>"}
  ],
  "reasoning": "<step-by-step deduction using subject, topic, and context; test and eliminate wrong options and double-check negations/math/code>",
  "answer_label": "<exact label of the correct choice, or its text if unlabeled>",
  "answer_text": "<exact text of the correct choice>",
  "confidence": "<high | medium | low>",
  "next": "<exact text of the Next/Continue/Submit button if visible, else empty string>"
}"""

VERIFY_PROMPT = """Double-check the draft answer below against the quiz screenshot (and OCR helper if provided).
Check specifically for common traps:
- Negations or qualifiers missed (NOT, EXCEPT, FALSE, LEAST, ONLY, ALWAYS)
- Misread numbers, signs, exponents, units, or code operators
- Confusing two options or swapping a label (e.g. reasoning picks B's text but label says A)

Draft extracted earlier:
- Subject / Topic: {subject} / {topic}
- Context: {context}
- Question: {question}
- Choices: {choices_json}
- Draft reasoning: {reasoning}
- Draft pick: label={answer_label!r}, text={answer_text!r} (confidence={confidence!r})

Reply with ONLY one JSON object:
{{
  "checked_trap": "<the key constraint, visual detail, or potential trap you verified>",
  "reasoning": "<concise verification confirming or correcting the answer>",
  "answer_label": "<verified label of the correct choice, or its text if unlabeled>",
  "answer_text": "<verified exact text of the correct choice>",
  "confidence": "<high | medium | low>"
}}"""

# ---------------------------------------------------------------------------
# LOGGING — console + timestamped file so bugs leave a paper trail
# ---------------------------------------------------------------------------

log = logging.getLogger("quizbot")
log.setLevel(logging.DEBUG)
_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
log.addHandler(_sh)
if LOG_FILE:
    try:
        _fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        _fh.setFormatter(_fmt)
        log.addHandler(_fh)
    except OSError as e:
        log.warning(f"[init] could not open log file {LOG_FILE!r}: {e}")

# Serializes hotkey presses so double-tapping cannot overlap two solves.
_solve_lock = threading.Lock()


def dump_config():
    """Log the full runtime config once at startup for comparing runs."""
    cfg = {
        "MODEL": MODEL, "MODEL_NUM_CTX": MODEL_NUM_CTX,
        "MODEL_NUM_PREDICT": MODEL_NUM_PREDICT, "IMG_WIDTH": IMG_WIDTH,
        "USE_OCR_HINT": USE_OCR_HINT, "VERIFY_UNCERTAIN": VERIFY_UNCERTAIN,
        "BOX_OFFSET_X": BOX_OFFSET_X, "NEXT_LABELS": NEXT_LABELS,
        "DRY_RUN": DRY_RUN, "NEXT_TRIES": NEXT_TRIES,
        "NEXT_DELAY": NEXT_DELAY, "NEXT_STRIP_TOP": NEXT_STRIP_TOP,
        "CLICK_SETTLE": CLICK_SETTLE, "MAX_BUTTON_CHARS": MAX_BUTTON_CHARS,
        "DEBUG_SAVE_MODE": DEBUG_SAVE_MODE, "DEBUG_SAVE_DIR": DEBUG_SAVE_DIR,
        "tesseract_cmd": pytesseract.pytesseract.tesseract_cmd,
        "python": sys.version.split()[0], "platform": platform.system(),
    }
    log.debug(f"[init] config: {json.dumps(cfg)}")
    return cfg


# ---------------------------------------------------------------------------
# STAGE 1 — SCREEN GRAB
# ---------------------------------------------------------------------------

def grab_screen():
    """Grab the primary monitor. Returns (PIL RGB image, monitor dict)."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # index 0 = virtual desktop; 1 = primary
        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    return img, monitor


# ---------------------------------------------------------------------------
# TEXT NORMALISATION + SHARED HELPERS
# ---------------------------------------------------------------------------

def clean(s):
    """Lowercase and keep only Unicode letters/digits for resilient matching."""
    return "".join(c for c in (s or "") if c.isalnum()).lower()


def center(box):
    """Centre point of an OCR box dict (left/top/right/bottom keys)."""
    return (box["left"] + box["right"]) // 2, (box["top"] + box["bottom"]) // 2


# ---------------------------------------------------------------------------
# STAGE 2 / STAGE 5 — OCR
# ---------------------------------------------------------------------------

def _lines_from(img, monitor, scale):
    """Run Tesseract word-level OCR on ONE prepared image in real screen coords."""
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    def box(i):
        left = monitor["left"] + data["left"][i] // scale
        top = monitor["top"] + data["top"][i] // scale
        right = monitor["left"] + (data["left"][i] + data["width"][i]) // scale
        bottom = monitor["top"] + (data["top"][i] + data["height"][i]) // scale
        return left, top, right, bottom

    grouped = {}
    n_noise = 0
    for i, word in enumerate(data["text"]):
        try:
            conf = int(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1
        if not word.strip() or conf <= 0:
            n_noise += 1
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        grouped.setdefault(key, []).append(i)

    lines = []
    for idxs in grouped.values():
        boxes = [box(i) for i in idxs]
        words = [{"text": data["text"][i], "left": b[0], "top": b[1],
                  "right": b[2], "bottom": b[3]} for i, b in zip(idxs, boxes)]
        lines.append({
            "text": " ".join(w["text"] for w in words),
            "left": min(b[0] for b in boxes),
            "top": min(b[1] for b in boxes),
            "right": max(b[2] for b in boxes),
            "bottom": max(b[3] for b in boxes),
            "words": words,
        })
    log.debug(f"[ocr] pass done: {len(lines)} lines, {n_noise} noise tokens dropped")
    return lines


def dedupe_lines(lines):
    """Drop duplicate lines produced by overlapping OCR passes."""
    kept = []
    for line in lines:
        dup = False
        lw = max(1, (line["right"] - line["left"]) * (line["bottom"] - line["top"]))
        for k in kept:
            if clean(k["text"]) != clean(line["text"]):
                continue
            ox = max(0, min(k["right"], line["right"]) - max(k["left"], line["left"]))
            oy = max(0, min(k["bottom"], line["bottom"]) - max(k["top"], line["top"]))
            kw = max(1, (k["right"] - k["left"]) * (k["bottom"] - k["top"]))
            if ox * oy >= 0.5 * min(kw, lw):
                dup = True
                break
        if not dup:
            kept.append(line)
    return kept


def ocr_lines(img, monitor, thorough=False):
    """OCR the screen. Fast mode = 1 pass; thorough = 4 passes + dedupe."""
    plain = ImageOps.grayscale(img)
    lines = _lines_from(plain, monitor, 1)
    if not thorough:
        return lines

    big = plain.resize((plain.width * 2, plain.height * 2), Image.LANCZOS)
    lines += _lines_from(big, monitor, 2)

    dark_text = big.point(lambda p: 255 if p > 150 else 0)
    lines += _lines_from(dark_text, monitor, 2)

    r, g, b = img.split()
    darkest = ImageChops.darker(ImageChops.darker(r, g), b)
    big_c = darkest.resize((darkest.width * 2, darkest.height * 2), Image.LANCZOS)
    light_text = big_c.point(lambda p: 0 if p > 170 else 255)
    lines += _lines_from(light_text, monitor, 2)

    before = len(lines)
    lines = dedupe_lines(lines)
    log.debug(f"[ocr] thorough: {before} raw lines -> {len(lines)} after dedupe")
    return lines


def format_ocr_hint(lines, max_chars=OCR_HINT_MAX_CHARS):
    """Group OCR lines by visual row so markers stay paired with their choice text."""
    if not lines:
        return ""
    ordered = sorted(lines, key=lambda l: ((l["top"] + l["bottom"]) // 2, l["left"]))
    rows = []
    for line in ordered:
        ymid = (line["top"] + line["bottom"]) // 2
        if rows and abs(ymid - rows[-1][0]) <= 10:
            rows[-1][1].append(line)
        else:
            rows.append([ymid, [line]])

    formatted = []
    for _, row_lines in rows:
        row_lines.sort(key=lambda l: l["left"])
        joined = " | ".join(l["text"].strip() for l in row_lines if l["text"].strip())
        if joined:
            formatted.append(joined)

    transcript = "\n".join(formatted)
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars].rstrip() + "\n[...OCR truncated...]"
    return transcript


def _bottom_strip_lines(img, monitor):
    """OCR just the bottom strip of the screen (all 4 thorough passes)."""
    h = img.height
    strip_img = img.crop((0, int(h * NEXT_STRIP_TOP), img.width, h))
    strip_monitor = dict(monitor, top=monitor["top"] + int(h * NEXT_STRIP_TOP))
    return ocr_lines(strip_img, strip_monitor, thorough=True)


# ---------------------------------------------------------------------------
# STAGE 3 — VISION MODEL QUERY & VERIFICATION
# ---------------------------------------------------------------------------

def parse_model_json(raw):
    """Extract the JSON object from the model's reply, tolerating stray chatter."""
    raw = (raw or "").strip()
    candidates = [raw]
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start:end + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError(f"no JSON object in model reply; raw starts with: {raw[:240]!r}")


def prepare_model_image(img):
    """Downscale screenshot to IMG_WIDTH (or keep if already smaller) and encode PNG."""
    target_w = min(img.width, IMG_WIDTH)
    if target_w < img.width:
        target_h = max(1, round(img.height * target_w / img.width))
        prepared = img.resize((target_w, target_h), Image.LANCZOS)
    else:
        prepared = img
    buf = io.BytesIO()
    prepared.save(buf, format="PNG")
    return prepared, buf.getvalue()


def _call_ollama_json(user_prompt, png, ocr_hint="", tag="model"):
    """Send a structured prompt + image to Ollama and return (parsed_dict, raw_text, dt)."""
    content = user_prompt
    if ocr_hint:
        content += (
            "\n\nFull-resolution OCR transcript (helper only — trust the screenshot "
            "for layout, diagrams, math symbols, and button labels):\n"
            f"```\n{ocr_hint}\n```"
        )

    t0 = time.perf_counter()
    resp = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content, "images": [png]},
        ],
        format="json",
        keep_alive=-1,
        options={
            "temperature": 0,
            "num_ctx": MODEL_NUM_CTX,
            "num_predict": MODEL_NUM_PREDICT,
        },
    )
    dt = time.perf_counter() - t0
    raw = resp["message"]["content"]
    data = parse_model_json(raw)
    log.debug(f"[{tag}] replied in {dt:.2f}s ({len(raw)} chars)")
    return data, raw, dt


def ask_model(png, ocr_hint=""):
    """Primary solve query on the vision model."""
    return _call_ollama_json(PROMPT, png, ocr_hint=ocr_hint, tag="model")


def verify_model_answer(png, draft, ocr_hint=""):
    """Second-pass self-verification for uncertain or self-contradictory answers."""
    prompt = VERIFY_PROMPT.format(
        subject=draft.get("subject", ""),
        topic=draft.get("topic", ""),
        context=draft.get("context", ""),
        question=draft.get("question", ""),
        choices_json=json.dumps(draft.get("choices", []), ensure_ascii=False),
        reasoning=draft.get("reasoning", ""),
        answer_label=draft.get("answer_label", ""),
        answer_text=draft.get("answer_text", ""),
        confidence=draft.get("confidence", ""),
    )
    return _call_ollama_json(prompt, png, ocr_hint=ocr_hint, tag="verify")


# ---------------------------------------------------------------------------
# STAGE 4 — CHOICE MATCHING & CROSS-CHECKING
# ---------------------------------------------------------------------------

_LABEL_PREFIX_RE = re.compile(r"^(?:option|choice|answer)\s*[:.\-)]?\s*", re.IGNORECASE)


def _strip_label_prefix(s):
    return _LABEL_PREFIX_RE.sub("", (s or "").strip())


def _match_by_label(choices, answer_label):
    """Return index of choice matching answer_label, or None."""
    raw = (answer_label or "").strip()
    if not raw:
        return None
    targets = [clean(raw), clean(_strip_label_prefix(raw))]
    targets = [t for t in targets if t]

    for target in targets:
        for idx, c in enumerate(choices):
            lbl = clean(c.get("label", ""))
            if lbl and lbl == target:
                return idx

    # Handle e.g. answer_label="B) Paris" when choice label is "B"
    for idx, c in enumerate(choices):
        lbl = (c.get("label", "") or "").strip()
        clbl = clean(lbl)
        if not clbl:
            continue
        for cand in (raw, _strip_label_prefix(raw)):
            ccand = clean(cand)
            if ccand.startswith(clbl) and len(cand) > len(lbl):
                sep = cand[len(lbl):len(lbl) + 1]
                if sep and not sep.isalnum():
                    return idx
    return None


def _match_by_text(choices, answer_text):
    """Return index of choice matching answer_text (exact first, then best substring)."""
    target = clean(answer_text)
    if not target:
        return None

    for idx, c in enumerate(choices):
        ct = clean(c.get("text", ""))
        if ct and ct == target:
            return idx

    # Substring fallback (prefer closest length so "5" doesn't overshadow "50")
    candidates = []
    for idx, c in enumerate(choices):
        ct = clean(c.get("text", ""))
        if not ct:
            continue
        if (len(target) >= 2 and target in ct) or (len(ct) >= 2 and ct in target):
            candidates.append((abs(len(ct) - len(target)), idx))
    if candidates:
        return min(candidates)[1]
    return None


def find_chosen(choices, answer_label, answer_text=""):
    """Resolve the model's answer onto one of its extracted choices.

    Returns a dict:
      {"choice": dict | None, "index": int | None, "method": str, "mismatch": bool}
    Cross-checks answer_label against answer_text so label slips are caught
    and logged loudly for bughunting.
    """
    if not choices:
        return {"choice": None, "index": None, "method": "no-choices", "mismatch": False}

    idx_lbl = _match_by_label(choices, answer_label)
    idx_txt = _match_by_text(choices, answer_text)

    if idx_lbl is not None and idx_txt is not None:
        if idx_lbl == idx_txt:
            return {"choice": choices[idx_lbl], "index": idx_lbl,
                    "method": "label+text", "mismatch": False}
        # Prefer answer_text on disagreement: small VLMs slip a letter more often
        # than they hallucinate the full wording of a different option.
        log.warning(
            f"[pick] MISMATCH: answer_label={answer_label!r} -> choice[{idx_lbl}] "
            f"({choices[idx_lbl].get('text')!r}), but answer_text={answer_text!r} -> "
            f"choice[{idx_txt}] ({choices[idx_txt].get('text')!r}); preferring text match"
        )
        return {"choice": choices[idx_txt], "index": idx_txt,
                "method": "mismatch-prefer-text", "mismatch": True}

    if idx_lbl is not None:
        if (answer_text or "").strip():
            log.debug(f"[pick] matched by label only (answer_text {answer_text!r} did not match)")
        return {"choice": choices[idx_lbl], "index": idx_lbl,
                "method": "label-only", "mismatch": False}

    if idx_txt is not None:
        log.info(f"[pick] recovered via answer_text (answer_label {answer_label!r} missed)")
        return {"choice": choices[idx_txt], "index": idx_txt,
                "method": "text-recovery", "mismatch": False}

    # Unlabeled fallback: model may have put the option text inside answer_label.
    idx_fallback = _match_by_text(choices, answer_label)
    if idx_fallback is not None:
        return {"choice": choices[idx_fallback], "index": idx_fallback,
                "method": "label-as-text", "mismatch": False}

    return {"choice": None, "index": None, "method": "unmatched", "mismatch": False}


# ---------------------------------------------------------------------------
# STAGE 5b — SPOTTING THE CHOICE ON SCREEN
# ---------------------------------------------------------------------------

def find_choice_spot(lines, choice, question):
    """Return ((x, y), strategy, matched_line) or (None, None, None)."""
    qc = clean(question)
    label = clean(choice.get("label", ""))
    text = clean(choice.get("text", ""))

    if label:
        exact_markers = [l for l in lines if clean(l["text"]) == label]
        if not exact_markers:
            max_len = len(label) + MAX_MARKER_EXTRA_CHARS
            exact_markers = [
                l for l in lines
                if clean(l["text"]).startswith(label) and len(clean(l["text"])) <= max_len
            ]
        if exact_markers:
            row = max(exact_markers, key=lambda l: l["top"])
            y = (row["top"] + row["bottom"]) // 2
            x = row["right"] + BOX_OFFSET_X
            log.debug(f"[locate] marker strategy: line {row['text']!r} -> ({x},{y})")
            return (x, y), "marker", row

    if text:
        limit = len(text) * CHOICE_LEN_FACTOR + CHOICE_LEN_SLACK
        hits = [
            l for l in lines
            if text in clean(l["text"])
            and clean(l["text"]) not in qc
            and len(clean(l["text"])) <= limit
        ]
        if hits:
            # Prefer exact text match (with or without label), then shortest line.
            best = min(
                hits,
                key=lambda l: (
                    0 if clean(l["text"]) in (text, label + text) else 1,
                    len(clean(l["text"])),
                ),
            )
            spot = center(best)
            log.debug(f"[locate] text strategy: {len(hits)} hits, chose {best['text']!r} -> {spot}")
            return spot, "text", best

    log.debug(f"[locate] no OCR line matched label={label!r} text={text!r}; "
              f"first 20 OCR lines were {[l['text'] for l in lines][:20]}")
    return None, None, None


# ---------------------------------------------------------------------------
# STAGE 6 — FINDING + CLICKING
# ---------------------------------------------------------------------------

def find_next_spot(lines, next_label):
    """Locate the Next button. Returns ((x, y), match_kind, matched_text) or (None, None, None)."""
    wanted = [clean(w) for w in [next_label] + NEXT_LABELS if clean(w)]

    for w in wanted:
        for line in lines:
            if clean(line["text"]) == w:
                return center(line), "exact-line", line["text"]

    for w in wanted:
        for line in lines:
            for word in line["words"]:
                if clean(word["text"]) == w:
                    return center(word), "exact-word", word["text"]

    fuzzy = []
    for w in wanted:
        for line in lines:
            c = clean(line["text"])
            if w in c and len(c) <= MAX_BUTTON_CHARS:
                fuzzy.append((len(c), center(line), "fuzzy-line", line["text"]))
        for line in lines:
            for word in line["words"]:
                c = clean(word["text"])
                if w in c and len(c) <= MAX_BUTTON_CHARS:
                    fuzzy.append((len(c), center(word), "fuzzy-word", word["text"]))
    if fuzzy:
        _, spot, kind, mtext = min(fuzzy)
        return spot, kind, mtext
    return None, None, None


def click(x, y, desc, dry_run=DRY_RUN):
    """Click via pyautogui, or log the coordinate in dry-run mode."""
    if dry_run:
        log.info(f"[dry run] would click {desc} at {(x, y)}")
    else:
        pyautogui.click(x, y)
        log.info(f"[click] clicked {desc} at {(x, y)}")


def click_next(next_label, dry_run=DRY_RUN):
    """Poll-retry finding and clicking the Next/Submit button after answering."""
    last_lines, monitor = [], None
    for attempt in range(NEXT_TRIES):
        img, monitor = grab_screen()
        spot, kind, mtext = find_next_spot(ocr_lines(img, monitor), next_label)
        if not spot:
            last_lines = ocr_lines(img, monitor, thorough=True)
            spot, kind, mtext = find_next_spot(last_lines, next_label)
        if not spot and attempt >= NEXT_TRIES // 2:
            bottom = _bottom_strip_lines(img, monitor)
            spot, kind, mtext = find_next_spot(bottom, next_label)
        if spot:
            log.debug(f"[next] attempt {attempt + 1}/{NEXT_TRIES} matched {kind} ({mtext!r}) -> {spot}")
            click(*spot, "Next", dry_run=dry_run)
            return {"ok": True, "spot": spot, "kind": kind, "text": mtext, "attempt": attempt + 1}
        log.debug(f"[next] attempt {attempt + 1}/{NEXT_TRIES}: button not found yet")
        if attempt < NEXT_TRIES - 1:
            time.sleep(NEXT_DELAY)

    third = monitor["height"] // 3 if monitor else 0
    top = monitor["top"] + third if monitor else 0
    bot = monitor["top"] + monitor["height"] - third if monitor else 0
    seen = [l["text"] for l in last_lines
            if (l["top"] < top or l["top"] >= bot) and len(l["text"]) <= 25]
    log.warning(f"[next] gave up. Short OCR text near top AND bottom of screen: {seen}")
    log.warning("[next] Click Next manually. If the button says something else, "
                "add that word to NEXT_LABELS.")
    return {"ok": False, "spot": None, "kind": None, "text": None, "seen_edges": seen}


# ---------------------------------------------------------------------------
# BUGHUNT ARTIFACTS — save annotated screenshot + full JSON trace
# ---------------------------------------------------------------------------

def _prune_debug_runs(base_dir):
    """Keep only the most recent DEBUG_KEEP_LAST run folders."""
    if DEBUG_KEEP_LAST <= 0 or not base_dir.exists():
        return
    runs = sorted(
        (p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("run_")),
        key=lambda p: p.name,
    )
    for old in runs[:-DEBUG_KEEP_LAST]:
        shutil.rmtree(old, ignore_errors=True)


def _draw_crosshair(draw, x, y, color, label, radius=14):
    """Draw a high-visibility target crosshair + label on the annotated image."""
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
    draw.line((x - radius - 8, y, x + radius + 8, y), fill=color, width=2)
    draw.line((x, y - radius - 8, x, y + radius + 8), fill=color, width=2)
    draw.text((x + radius + 6, y - 8), label, fill=color)


def _save_debug_bundle(trace, full_img, model_img, monitor, ocr_sample_lines,
                       choice_spot, choice_box, next_spot):
    """Write model_input.png, annotated.png, and trace.json for post-mortem bughunting."""
    mode = (DEBUG_SAVE_MODE or "off").lower()
    if mode == "off":
        return None
    if mode == "on_error" and not trace.get("error") and trace.get("pick", {}).get("index") is not None:
        return None

    base_dir = Path(DEBUG_SAVE_DIR)
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")[:-3]
    run_dir = base_dir / stamp
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        if model_img is not None:
            model_img.save(run_dir / "model_input.png")

        if full_img is not None:
            annotated = full_img.copy()
            draw = ImageDraw.Draw(annotated)
            ox = monitor["left"] if monitor else 0
            oy = monitor["top"] if monitor else 0

            for line in (ocr_sample_lines or [])[:120]:
                draw.rectangle(
                    (line["left"] - ox, line["top"] - oy, line["right"] - ox, line["bottom"] - oy),
                    outline=(255, 215, 0), width=1,
                )
            if choice_box:
                draw.rectangle(
                    (choice_box["left"] - ox, choice_box["top"] - oy,
                     choice_box["right"] - ox, choice_box["bottom"] - oy),
                    outline=(0, 220, 100), width=3,
                )
            if choice_spot:
                _draw_crosshair(draw, choice_spot[0] - ox, choice_spot[1] - oy,
                                (0, 220, 100), f"CHOICE {choice_spot}")
            if next_spot:
                _draw_crosshair(draw, next_spot[0] - ox, next_spot[1] - oy,
                                (0, 160, 255), f"NEXT {next_spot}")
            annotated.save(run_dir / "annotated.png")

        (run_dir / "trace.json").write_text(
            json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _prune_debug_runs(base_dir)
        log.debug(f"[debug] saved run bundle -> {run_dir}")
        return str(run_dir)
    except Exception as e:
        log.warning(f"[debug] failed to save debug bundle: {e}")
        return None


# ---------------------------------------------------------------------------
# STAGE 7 — FULL SOLVE CYCLE (live screen or offline --test image)
# ---------------------------------------------------------------------------

def _log_model_understanding(data):
    """Print subject, topic, context, parsed question/choices, and reasoning."""
    subject = data.get("subject", "") or "-"
    topic = data.get("topic", "") or "-"
    context = data.get("context", "") or "-"
    confidence = data.get("confidence", "") or "-"
    log.info(f"[context] Subject: {subject} | Topic: {topic} | Confidence: {confidence}")
    log.info(f"[context] Screen context: {context}")
    log.info(f"[parse] Q: {data.get('question', '')}")
    for c in data.get("choices", []):
        if isinstance(c, dict):
            log.info(f"[parse]   {c.get('label') or '-'}: {c.get('text')}")
    log.info(f"[reason] {data.get('reasoning', '')}")


def _maybe_verify(png, data, pick_res, ocr_hint, allow_verify, trace):
    """Run a targeted self-check when confidence is low/medium or label/text disagreed."""
    if not (allow_verify and VERIFY_UNCERTAIN):
        return data, pick_res
    conf = str(data.get("confidence", "")).strip().lower()
    if conf == "high" and not pick_res["mismatch"]:
        return data, pick_res

    reason = "label/text mismatch" if pick_res["mismatch"] else f"confidence={conf or 'unknown'!r}"
    log.info(f"[verify] running second-pass check ({reason})...")
    v_data, v_raw, v_dt = verify_model_answer(png, data, ocr_hint=ocr_hint)
    trace["verify"] = {"raw": v_raw, "parsed": v_data, "seconds": round(v_dt, 3)}
    log.info(f"[verify] trap checked: {v_data.get('checked_trap', '')}")
    log.info(f"[verify] reasoning: {v_data.get('reasoning', '')}")

    v_pick = find_chosen(
        data.get("choices", []),
        v_data.get("answer_label", ""),
        v_data.get("answer_text", ""),
    )
    if v_pick["choice"] is not None:
        merged = dict(data)
        for key in ("reasoning", "answer_label", "answer_text", "confidence"):
            if v_data.get(key):
                merged[key] = v_data[key]
        return merged, v_pick
    log.warning("[verify] verification answer did not match choices; keeping primary pick")
    return data, pick_res


def solve_image(img, monitor, dry_run=DRY_RUN, allow_verify=True, static_image=False):
    """Run one complete solve cycle on `img` (live capture or offline --test image)."""
    t_total = time.perf_counter()
    trace = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "static_image": static_image,
        "monitor": monitor,
    }
    model_img = None
    ocr_fast = []
    choice_spot = None
    choice_box = None
    next_spot = None

    try:
        t_ocr0 = time.perf_counter()
        ocr_fast = ocr_lines(img, monitor)
        t_ocr_fast = time.perf_counter() - t_ocr0
        ocr_hint = format_ocr_hint(ocr_fast) if USE_OCR_HINT else ""

        model_img, png = prepare_model_image(img)
        log.debug(
            f"[model] sending {model_img.width}x{model_img.height} PNG "
            f"({len(png) // 1024} KiB), ocr_hint={len(ocr_hint)} chars"
        )
        data, raw_reply, t_model = ask_model(png, ocr_hint=ocr_hint)

        trace["ocr_hint"] = ocr_hint
        trace["model"] = {"raw": raw_reply, "parsed": data, "seconds": round(t_model, 3)}

        choices = data.get("choices", [])
        if not isinstance(choices, list) or not choices:
            _log_model_understanding(data)
            log.error("[pick] model returned no choices — cannot answer. "
                      f"Raw keys: {list(data)}")
            trace["error"] = "no choices returned"
            return

        _log_model_understanding(data)

        pick_res = find_chosen(
            choices,
            data.get("answer_label", ""),
            data.get("answer_text", ""),
        )
        data, pick_res = _maybe_verify(png, data, pick_res, ocr_hint, allow_verify, trace)
        trace["pick"] = {
            "index": pick_res["index"],
            "method": pick_res["method"],
            "mismatch": pick_res["mismatch"],
            "choice": pick_res["choice"],
        }

        chosen = pick_res["choice"]
        if not chosen:
            log.error(
                f"[pick] answer_label={data.get('answer_label', '')!r}, "
                f"answer_text={data.get('answer_text', '')!r} matched none of the choices. "
                "Skipping (not guessing)."
            )
            trace["error"] = "pick unmatched"
            return

        log.info(
            f"[pick] chose [{pick_res['method']}]: "
            f"{chosen.get('label') or '-'} -> {chosen.get('text')}"
        )

        t_extra0 = time.perf_counter()
        choice_spot, strategy, choice_box = find_choice_spot(
            ocr_fast, chosen, data.get("question", "")
        )
        ocr_all = ocr_fast
        if not choice_spot:
            log.debug("[locate] fast OCR missed, retrying with thorough (4-pass) OCR")
            ocr_all = ocr_lines(img, monitor, thorough=True)
            choice_spot, strategy, choice_box = find_choice_spot(
                ocr_all, chosen, data.get("question", "")
            )
        t_ocr_extra = time.perf_counter() - t_extra0
        trace["locate"] = {
            "spot": choice_spot,
            "strategy": strategy,
            "matched_text": choice_box["text"] if choice_box else None,
        }

        if not choice_spot:
            log.error("[locate] could not find the chosen option on screen even after "
                      "thorough OCR. Check BOX_OFFSET_X / CHOICE_LEN_* in CONFIG.")
            trace["error"] = "locate failed"
            return

        click(*choice_spot, "choice", dry_run=dry_run)

        t_next0 = time.perf_counter()
        if static_image:
            # In offline --test mode, inspect the static image once without polling.
            next_spot, next_kind, next_text = find_next_spot(ocr_all, data.get("next", "") or "Next")
            if not next_spot:
                next_spot, next_kind, next_text = find_next_spot(
                    _bottom_strip_lines(img, monitor), data.get("next", "") or "Next"
                )
            next_res = {"ok": bool(next_spot), "spot": next_spot, "kind": next_kind, "text": next_text}
            if next_spot:
                log.info(f"[next] (static check) matched {next_kind} ({next_text!r}) -> {next_spot}")
            else:
                log.info("[next] (static check) no Next button visible in test image")
        else:
            pyautogui.sleep(CLICK_SETTLE)
            next_res = click_next(data.get("next", "") or "Next", dry_run=dry_run)
            next_spot = next_res.get("spot")
        t_next = time.perf_counter() - t_next0
        trace["next"] = next_res

        t_total_elapsed = time.perf_counter() - t_total
        trace["timing"] = {
            "ocr_fast": round(t_ocr_fast, 3),
            "model": round(t_model, 3),
            "ocr_extra": round(t_ocr_extra, 3),
            "click_next": round(t_next, 3),
            "total": round(t_total_elapsed, 3),
        }
        log.info(
            f"[time] ocr-fast {t_ocr_fast:.2f}s | model {t_model:.2f}s | "
            f"ocr-extra {t_ocr_extra:.2f}s | click+next {t_next:.2f}s | "
            f"total {t_total_elapsed:.2f}s | next={'ok' if next_res.get('ok') else 'FAILED'}"
        )
    except Exception as e:
        trace["error"] = f"{type(e).__name__}: {e}"
        trace["traceback"] = traceback.format_exc()
        log.exception(f"[error] solve cycle failed: {e}")
    finally:
        _save_debug_bundle(
            trace, img, model_img, monitor, ocr_fast,
            choice_spot, choice_box, next_spot,
        )


def solve_one(dry_run=DRY_RUN, allow_verify=True):
    """Grab primary screen and run solve_image."""
    t0 = time.perf_counter()
    img, monitor = grab_screen()
    log.debug(
        f"[grab] {img.width}x{img.height} monitor@({monitor['left']},{monitor['top']}) "
        f"in {(time.perf_counter() - t0) * 1000:.0f} ms"
    )
    solve_image(img, monitor, dry_run=dry_run, allow_verify=allow_verify, static_image=False)


def _locked_solve(dry_run, allow_verify):
    """Worker body: run one solve, always release the lock at the end."""
    try:
        solve_one(dry_run=dry_run, allow_verify=allow_verify)
    finally:
        _solve_lock.release()


def make_hotkey_handler(dry_run=DRY_RUN, allow_verify=True):
    """Build the Ctrl+Alt+Q callback with runtime flags bound."""
    def on_hotkey():
        if not _solve_lock.acquire(blocking=False):
            log.warning("[hotkey] a solve is already running — press ignored.")
            return
        threading.Thread(
            target=_locked_solve, args=(dry_run, allow_verify), daemon=True
        ).start()
        log.info("[hotkey] starting solve cycle...")
    return on_hotkey


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Local screen-reading quiz solver.")
    parser.add_argument(
        "--test", "--replay", dest="test_image", metavar="IMAGE",
        help="Run once offline on a saved screenshot (implies --dry-run, no clicks).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log target coordinates instead of clicking.",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip the second-pass self-check on low/medium confidence answers.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    dry_run = DRY_RUN or args.dry_run or bool(args.test_image)
    allow_verify = not args.no_verify
    dump_config()

    if args.test_image:
        test_path = Path(args.test_image)
        if not test_path.is_file():
            log.error(f"[test] image not found: {test_path}")
            sys.exit(1)
        log.info(f"[test] running offline dry-run on {test_path}")
        with Image.open(test_path) as loaded:
            rgb_img = loaded.convert("RGB")
        fake_monitor = {"left": 0, "top": 0, "width": rgb_img.width, "height": rgb_img.height}
        solve_image(
            rgb_img, fake_monitor,
            dry_run=True, allow_verify=allow_verify, static_image=True,
        )
        sys.exit(0)

    mode = "DRY RUN (no clicks)" if dry_run else "LIVE"
    log.info(f"Ready in {mode} mode. Click the quiz window, then press Ctrl+Alt+Q.")
    log.info("Slam the mouse into a screen corner to stop (pyautogui failsafe).")
    log.info(f"Detailed logs -> {LOG_FILE or 'console only'} | Debug bundles -> {DEBUG_SAVE_DIR}/")
    handler = make_hotkey_handler(dry_run=dry_run, allow_verify=allow_verify)
    with keyboard.GlobalHotKeys({"<ctrl>+<alt>+q": handler}) as listener:
        listener.join()
