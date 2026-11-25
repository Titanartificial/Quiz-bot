"""
QUIZ-BOT — local screen-reading quiz solver.

HOW THE PIPELINE WORKS (read this before bugfixing!)
====================================================

Each hotkey press runs one full "solve cycle". The cycle has 6 stages, and every
stage prints a [tag] so you can tell exactly where things went wrong:

  [grab]   Stage 1 — Screenshot the primary monitor with mss (fast, raw pixels).
  [model]  Stage 2 — Downscale the image and ask the vision model (Ollama) to
                     extract {question, choices, answer_label, next} as JSON.
                     The MODEL IS UNCHANGED: same model name, same prompt, same
                     sampling options (temperature 0, num_predict 400, keep_alive).
  [pick]   Stage 3 — Match the model's answer_label back to one of its own
                     choices (labels first, then text). Fuzzy matching rules are
                     documented in find_chosen().
  [locate] Stage 4 — Tesseract OCR finds WHERE that choice is on the real screen.
                     Two passes: fast (1 grayscale pass), then thorough (4 passes)
                     only if the fast pass fails. See ocr_lines() for why each
                     pre-processing pass exists. Matching rules documented in
                     find_choice_spot().
  [click]  Stage 5 — pyautogui clicks the found coordinate (or prints it in dry
                     run mode). Then click_next() hunts for the Next button with
                     retries, because it only appears after the tick registers.
  [time]   Stage 6 — Per-stage timing summary so you can see what is slow.

DEBUGGING CHEAT-SHEET
=====================
* "[model] ... no JSON object"  -> model returned prose instead of JSON; the raw
                                   reply snippet is dumped to the log & console.
* "[pick] no match"             -> model's answer_label doesn't correspond to any
                                   choice it listed; both are printed.
* "[locate] not found (fast)"   -> normal; it automatically retries in thorough
                                   mode. Only a real failure if thorough also fails.
* "[locate] marker hit but wrong spot" -> tune BOX_OFFSET_X.
* "[next] gave up"              -> prints short OCR lines near the top of the
                                   screen so you can see what the button actually
                                   says; add that word to NEXT_LABELS.
* Everything is also appended to quizbot.log with timestamps + a full config dump
  at startup, so you can diff two runs.

Dependencies: mss, pytesseract (+Tesseract binary), pyautogui, Pillow, pynput,
ollama (with MODEL pulled). No changes to the model itself.
"""

import io
import json
import logging
import sys
import platform
import threading
import time

import mss
import pytesseract
import pyautogui
from PIL import Image, ImageChops, ImageOps
from pynput import keyboard
import ollama

# ---------------------------------------------------------------------------
# CONFIG — everything tunable lives here
# ---------------------------------------------------------------------------

MODEL = "qwen2.5vl:3b"      # exact same model as before — do not treat as tunable
BOX_OFFSET_X = 40           # px from a lone marker like "A" to the tickbox right of it.
NEXT_LABELS = ["Next", "Continue", "Submit", "Finish", "Done"]  # fallback button words
DRY_RUN = False             # True = print coordinates instead of clicking.
IMG_WIDTH = 768             # screenshot width sent to the model (smaller = faster).
NEXT_TRIES = 4              # attempts to find the Next button (it appears late).
NEXT_DELAY = 0.8            # seconds between Next-button retry screenshots.
CLICK_SETTLE = 0.5          # pause after ticking an answer, before hunting Next.
LOG_FILE = "quizbot.log"    # set to None to disable file logging.

# Tunables for the OCR matching heuristics (see find_choice_spot):
MAX_MARKER_EXTRA_CHARS = 1  # "A)" cleans to "a" + ")" -> allow 1 stray char.
CHOICE_LEN_FACTOR = 2       # OCR line may be at most factor * choice length...
CHOICE_LEN_SLACK = 6        # ...plus this many chars of slack (length cap).

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pyautogui.FAILSAFE = True   # slam the mouse into a corner to abort everything.
pyautogui.PAUSE = 0.05      # tiny gap between every GUI action (stability).

PROMPT = """This is a screenshot of a quiz. Do three things:
1. Extract the current question and every answer choice. Quizzes format choices in different ways (letters, numbers, bullets, radio circles, checkboxes, or no marker), so copy whatever marker is shown.
2. Decide which choice is correct.
3. Find the button that moves to the next question (like Next or Continue). Use an empty string if none is visible.
Reply with ONLY a JSON object and no other text:
{"question": "<question text>",
 "choices": [{"label": "<marker exactly as shown, or empty string>", "text": "<choice text>"}],
 "answer_label": "<label of the correct choice, or its text if it has no label>",
 "next": "<exact text of the next button, or empty string>"}"""

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
    except OSError as e:  # read-only folder etc — console logging still works
        log.warning(f"[init] could not open log file {LOG_FILE!r}: {e}")

# A lock so double-tapping the hotkey can't start two overlapping solves
# (which would double-click the same question and desync everything).
_solve_lock = threading.Lock()


def dump_config():
    """Log the full runtime config once at startup — useful when comparing runs."""
    cfg = {
        "MODEL": MODEL, "BOX_OFFSET_X": BOX_OFFSET_X, "NEXT_LABELS": NEXT_LABELS,
        "DRY_RUN": DRY_RUN, "IMG_WIDTH": IMG_WIDTH, "NEXT_TRIES": NEXT_TRIES,
        "NEXT_DELAY": NEXT_DELAY, "CLICK_SETTLE": CLICK_SETTLE,
        "tesseract_cmd": pytesseract.pytesseract.tesseract_cmd,
        "python": sys.version.split()[0], "platform": platform.system(),
    }
    log.debug(f"[init] config: {json.dumps(cfg)}")


# ---------------------------------------------------------------------------
# STAGE 1 — SCREEN GRAB
# ---------------------------------------------------------------------------

def grab_screen():
    """Grab the primary monitor. Returns (PIL RGB image, monitor dict).

    monitor holds the real-screen origin (left/top) + size, needed to convert
    OCR pixel offsets into absolute click coordinates. mss is much faster than
    pyautogui.screenshot; it returns BGRA bytes which we reinterpret directly
    ("BGRX" raw mode swaps R/B so the colours come out correct — keep it).
    """
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # index 0 is the whole virtual desktop; 1 = primary
        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    return img, monitor


# ---------------------------------------------------------------------------
# STAGE 2 — VISION MODEL QUERY  (model + prompt identical to the original)
# ---------------------------------------------------------------------------

def parse_model_json(raw):
    """Extract the JSON object from the model's reply.

    We asked for format="json", so normally this is a straight json.loads.
    But small models occasionally wrap output in ``` fences or add chatter,
    and the old code crashed with a bare json.JSONDecodeError. Strategy:
      1. try json.loads on the whole string;
      2. else slice out the outermost {...} braces and try again;
      3. else raise ValueError including a snippet of the raw reply so the
         log tells you WHAT the model actually said.
    """
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
    raise ValueError(f"no JSON object in model reply; raw starts with: {raw[:200]!r}")


def ask_model(img):
    """Downscale -> PNG bytes -> Ollama chat -> parsed dict. Raises on failure.

    Downsizing to IMG_WIDTH keeps the payload small (big speed win) while
    staying legible for a vision model. temperature=0 makes replies repeatable,
    keep_alive=-1 pins the model in RAM/VRAM so later calls skip reload latency.
    """
    small = img.resize((IMG_WIDTH, max(1, round(img.height * IMG_WIDTH / img.width))),
                       Image.LANCZOS)
    buf = io.BytesIO()
    small.save(buf, format="PNG")
    png = buf.getvalue()
    log.debug(f"[model] sending {small.width}x{small.height} PNG ({len(png) // 1024} KiB)")

    t0 = time.perf_counter()
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT, "images": [png]}],
        format="json",
        keep_alive=-1,
        options={"temperature": 0, "num_predict": 400},
    )
    dt = time.perf_counter() - t0
    raw = resp["message"]["content"]
    data = parse_model_json(raw)
    log.debug(f"[model] replied in {dt:.1f}s: {json.dumps(data, ensure_ascii=False)[:400]}")
    return data


# ---------------------------------------------------------------------------
# TEXT NORMALISATION + SHARED MATCHING HELPERS
# ---------------------------------------------------------------------------

def clean(s):
    """Lowercase and keep only letters/digits, so 'A)', 'a.' and 'A' compare equal.

    Unicode-aware by design: accented letters count as alnum (str.isalnum),
    which matters for non-English quizzes where ASCII-only stripping would
    silently destroy the text and make every match fail.
    """
    return "".join(c for c in (s or "") if c.isalnum()).lower()


def center(box):
    """Centre point of an OCR box dict (left/top/right/bottom keys)."""
    return (box["left"] + box["right"]) // 2, (box["top"] + box["bottom"]) // 2


# ---------------------------------------------------------------------------
# STAGE 4 — OCR
# ---------------------------------------------------------------------------

def _lines_from(img, monitor, scale):
    """Run Tesseract word-level data on ONE prepared image.

    Returns line dicts in REAL SCREEN coordinates:
      {"text", "left", "top", "right", "bottom", "words": [word boxes...]}

    `scale` is how much bigger `img` is than the real screen (1 or 2); we divide
    coordinates back down and add the monitor origin so results map to pyautogui.
    Words are grouped into lines by Tesseract's block/para/line numbering.
    Word boxes are kept because find_next_spot needs single-word matches when
    OCR merges a button with neighbouring text ("A Correct! NEXT").
    Tokens with conf <= 0 are Tesseract noise markers and get dropped early —
    that stops junk words from polluting line text used for matching.
    """
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
    """Drop duplicate lines produced by overlapping OCR passes.

    Thorough mode runs 4 passes over the same screen; without dedupe, a button
    found by two passes yields two candidate hits and wastes work. Two lines
    count as duplicates when their cleaned text matches AND their boxes overlap
    by >=50% of the smaller area (position matters too: repeated 'A' markers
    stack vertically and must NOT collapse into one).
    """
    kept = []
    for l in lines:
        dup = False
        lw = max(1, (l["right"] - l["left"]) * (l["bottom"] - l["top"]))
        for k in kept:
            if clean(k["text"]) != clean(l["text"]):
                continue
            ox = max(0, min(k["right"], l["right"]) - max(k["left"], l["left"]))
            oy = max(0, min(k["bottom"], l["bottom"]) - max(k["top"], l["top"]))
            kw = max(1, (k["right"] - k["left"]) * (k["bottom"] - k["top"]))
            if ox * oy >= 0.5 * min(kw, lw):
                dup = True
                break
        if not dup:
            kept.append(l)
    return kept


def ocr_lines(img, monitor, thorough=False):
    """OCR the screen. Fast mode = 1 pass; thorough = 4 passes + dedupe.

    Why each extra pass exists (this is the usual culprit when [locate] fails):
      * plain grayscale ......... baseline; catches normal-sized dark-on-light UI
      * 2x upscale .............. small fonts under ~12px fall below Tesseract's
                                  accuracy floor; upscaling fixes them
      * thresholded bright bg ... crisp dark text on white (buttons like "NEXT")
      * inverted chroma ......... LIGHT text on coloured buttons: take the darkest
                                  channel per pixel (coloured bg -> mid, white
                                  text -> high), threshold + invert so the text
                                  becomes black-on-white for Tesseract.
    """
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


# ---------------------------------------------------------------------------
# STAGE 3 — CHOICE MATCHING (model answer -> one of the model's choices)
# ---------------------------------------------------------------------------

def find_chosen(choices, pick):
    """Map the model's answer_label onto one of its own choices.

    Rules, in priority order (same semantics as the original, now documented):
      1. exact cleaned-label match   ("A)" vs "a" -> equal after clean())
      2. exact cleaned-text match    (model echoed the full choice text)
      3. substring text match        (model answered a shortened version)
    Returns the choice dict or None. Empty/whitespace picks never match, so a
    hallucinated "" answer cannot accidentally click option 0.
    """
    p = clean(pick)
    if not p:
        return None
    for c in choices:
        if clean(c.get("label", "")) == p:
            return c
    for c in choices:
        ct = clean(c.get("text", ""))
        if ct and (p == ct or p in ct):
            return c
    return None


# ---------------------------------------------------------------------------
# STAGE 4b — SPOTTING THE CHOICE ON SCREEN
# ---------------------------------------------------------------------------

def find_choice_spot(lines, choice, question):
    """Return (x, y) to click for this choice, or None. Two strategies:

    A) MARKER STRATEGY — there is an OCR line that is just the marker ("A", "1",
       "A)"). Click BOX_OFFSET_X px to the RIGHT of the marker's box, vertically
       centred on that line — that's where radio buttons/tickboxes sit.
       If several lines match the marker (e.g. stray OCR hits of a bare letter),
       prefer the LOWEST one: answer lists run top-to-bottom, and spurious
       single-letter detections usually come from elsewhere on the page.
    B) TEXT STRATEGY — no usable marker line. Find a SHORT line containing the
       choice text. The length cap (factor + slack, see CONFIG) stops us from
       clicking the question paragraph when it happens to quote an option;
       lines whose text is part of the question are excluded outright. Among
       hits, the SHORTEST line wins (least likely to be a sentence that merely
       contains the phrase). Click its centre.
    On failure, logs the candidate OCR lines so you can see WHY nothing matched.
    """
    qc = clean(question)
    label = clean(choice.get("label", ""))
    text = clean(choice.get("text", ""))

    if label:
        max_len = len(label) + MAX_MARKER_EXTRA_CHARS
        markers = [l for l in lines
                   if clean(l["text"]) == label
                   or (clean(l["text"]).startswith(label)
                       and len(clean(l["text"])) <= max_len)]
        if markers:
            row = max(markers, key=lambda l: l["top"])
            y = (row["top"] + row["bottom"]) // 2
            x = row["right"] + BOX_OFFSET_X
            log.debug(f"[locate] marker strategy: line {row['text']!r} -> ({x},{y})")
            return x, y

    if text:
        limit = len(text) * CHOICE_LEN_FACTOR + CHOICE_LEN_SLACK
        hits = [l for l in lines
                if text in clean(l["text"])
                and clean(l["text"]) not in qc
                and len(clean(l["text"])) <= limit]
        if hits:
            best = min(hits, key=lambda l: len(clean(l["text"])))
            log.debug(f"[locate] text strategy: {len(hits)} hits, chose {best['text']!r}")
            return center(best)

    log.debug(f"[locate] no OCR line matched label={label!r} text={text!r}; "
              f"first 20 OCR lines were {[l['text'] for l in lines][:20]}")
    return None


# ---------------------------------------------------------------------------
# STAGE 5 — FINDING + CLICKING
# ---------------------------------------------------------------------------

def find_next_spot(lines, next_label):
    """Locate the Next button. Checks in priority order:
      1. the model-reported label, then the NEXT_LABELS fallbacks —
         a WHOLE LINE whose cleaned text equals it.
      2. failing that, a single WORD inside any line equal to it, because OCR
         often merges the button with nearby text ("A Correct! NEXT").
    Returns (x, y) or None.
    """
    wanted = [clean(w) for w in [next_label] + NEXT_LABELS if w]

    for w in wanted:
        for line in lines:
            if clean(line["text"]) == w:
                return center(line)

    for w in wanted:
        for line in lines:
            for word in line["words"]:
                if clean(word["text"]) == w:
                    return center(word)
    return None


def click(x, y, desc):
    """Click via pyautogui, or just report the coordinate in DRY_RUN mode."""
    if DRY_RUN:
        log.info(f"[dry run] would click {desc} at {(x, y)}")
    else:
        pyautogui.click(x, y)
        log.info(f"[click] clicked {desc} at {(x, y)}")


def click_next(next_label):
    """The Next button only renders AFTER our tick registers, so poll-retry:
    each attempt takes a fresh screenshot, tries fast OCR, then thorough OCR.
    Waits NEXT_DELAY between attempts. On final failure, dumps short OCR lines
    from the top third of the screen (where headers/buttons usually live) so
    you can see what the button actually says and extend NEXT_LABELS.
    """
    last_lines, monitor = [], None
    for attempt in range(NEXT_TRIES):
        img, monitor = grab_screen()
        spot = find_next_spot(ocr_lines(img, monitor), next_label)
        if not spot:
            last_lines = ocr_lines(img, monitor, thorough=True)
            spot = find_next_spot(last_lines, next_label)
        if spot:
            click(*spot, "Next")
            return True
        log.debug(f"[next] attempt {attempt + 1}/{NEXT_TRIES}: button not found yet")
        if attempt < NEXT_TRIES - 1:
            time.sleep(NEXT_DELAY)

    top = monitor["top"] + monitor["height"] // 3 if monitor else 0
    seen = [l["text"] for l in last_lines if l["top"] < top and len(l["text"]) <= 25]
    log.warning(f"[next] gave up. Short OCR text near top of screen: {seen}")
    log.warning("[next] Click Next manually. If the button says something else, "
                "add that word to NEXT_LABELS.")
    return False


# ---------------------------------------------------------------------------
# STAGE 6 — FULL SOLVE CYCLE
# ---------------------------------------------------------------------------

def solve_one():
    """One question: grab -> model -> pick -> locate -> click -> next.

    Every stage times itself; the [time] summary line tells you where the
    seconds go. Any exception is caught and logged WITH traceback so the
    hotkey listener survives and you get a stack trace in quizbot.log.
    """
    t_total = time.perf_counter()
    try:
        img, monitor = grab_screen()
        t_grab = time.perf_counter()
        log.debug(f"[grab] {img.width}x{img.height} monitor@({monitor['left']},"
                  f"{monitor['top']}) in {(t_grab - t_total) * 1000:.0f} ms")

        data = ask_model(img)
        t_model = time.perf_counter()

        question = data.get("question", "")
        choices = data.get("choices", [])
        if not isinstance(choices, list) or not choices:
            log.error("[pick] model returned no choices — cannot answer. "
                      f"Raw data keys: {list(data)}")
            return

        log.info(f"[parse] Q: {question}")
        for c in choices:
            log.info(f"[parse]   {c.get('label') or '-'}: {c.get('text')}")

        chosen = find_chosen(choices, data.get("answer_label", ""))
        if not chosen:
            log.error(f"[pick] model's answer_label {data.get('answer_label', '')!r} "
                      "matched none of its own choices. Skipping (not guessing).")
            return
        log.info(f"[pick] chose: {chosen.get('label') or chosen.get('text')}")

        lines = ocr_lines(img, monitor)
        t_fast = time.perf_counter()
        spot = find_choice_spot(lines, chosen, question)
        if not spot:
            log.debug("[locate] fast OCR missed, retrying with thorough (4-pass) OCR")
            lines = ocr_lines(img, monitor, thorough=True)
            spot = find_choice_spot(lines, chosen, question)
        t_ocr = time.perf_counter()
        if not spot:
            log.error("[locate] could not find the chosen option on screen even after "
                      "thorough OCR. Check BOX_OFFSET_X / the length caps in CONFIG.")
            return
        click(*spot, "choice")

        pyautogui.sleep(CLICK_SETTLE)
        ok_next = click_next(data.get("next", "") or "Next")
        t_end = time.perf_counter()

        log.info(f"[time] grab {(t_grab - t_total):.2f}s | model {(t_model - t_grab):.2f}s | "
                 f"ocr-fast {(t_fast - t_model):.2f}s | "
                 f"ocr-extra {(t_ocr - t_fast):.2f}s | "
                 f"click+next {(t_end - t_ocr):.2f}s | total {(t_end - t_total):.2f}s | "
                 f"next={'ok' if ok_next else 'FAILED'}")
    except Exception as e:
        log.exception(f"[error] solve cycle failed: {e}")


def _locked_solve():
    """Worker body: run one solve, always release the lock at the end."""
    try:
        solve_one()
    finally:
        _solve_lock.release()


def on_hotkey():
    """Hotkey callback. Runs the solve in a worker thread so the keyboard
    listener stays responsive; the non-blocking lock means a second press
    during an active solve is ignored instead of double-clicking."""
    if not _solve_lock.acquire(blocking=False):
        log.warning("[hotkey] a solve is already running — press ignored.")
        return
    threading.Thread(target=_locked_solve, daemon=True).start()
    log.info("[hotkey] starting solve cycle...")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dump_config()
    mode = "DRY RUN (no clicks)" if DRY_RUN else "LIVE"
    log.info(f"Ready in {mode} mode. Click the quiz window, then press Ctrl+Alt+Q.")
    log.info("Slam the mouse into a screen corner to stop (pyautogui failsafe).")
    log.info(f"Detailed logs -> {LOG_FILE or 'console only'}")
    with keyboard.GlobalHotKeys({"<ctrl>+<alt>+q": on_hotkey}) as listener:
        listener.join()
