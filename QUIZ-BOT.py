import io
import json
import threading
import time

import mss
import pytesseract
import pyautogui
from PIL import Image, ImageChops, ImageOps
from pynput import keyboard
import ollama

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pyautogui.FAILSAFE = True  # slam the mouse into a corner to abort
pyautogui.PAUSE = 0.05

MODEL = "qwen2.5vl:3b"
BOX_OFFSET_X = 40                # distance from a marker like "A" to its tickbox
NEXT_LABELS = ["Next", "Continue", "Submit", "Finish", "Done"]
DRY_RUN = False                  # True = print where it would click, without clicking
IMG_WIDTH = 768                  # screenshot width sent to the model
NEXT_TRIES = 4
NEXT_DELAY = 0.8

PROMPT = """This is a screenshot of a quiz. Do three things:
1. Extract the current question and every answer choice. Quizzes format choices in different ways (letters, numbers, bullets, radio circles, checkboxes, or no marker), so copy whatever marker is shown.
2. Decide which choice is correct.
3. Find the button that moves to the next question (like Next or Continue). Use an empty string if none is visible.
Reply with ONLY a JSON object and no other text:
{"question": "<question text>",
 "choices": [{"label": "<marker exactly as shown, or empty string>", "text": "<choice text>"}],
 "answer_label": "<label of the correct choice, or its text if it has no label>",
 "next": "<exact text of the next button, or empty string>"}"""


def grab_screen():
    with mss.MSS() as sct:
        monitor = sct.monitors[1]  # primary screen
        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    return img, monitor


def ask_model(img):
    small = img.resize((IMG_WIDTH, int(img.height * IMG_WIDTH / img.width)))
    buf = io.BytesIO()
    small.save(buf, format="PNG")
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT, "images": [buf.getvalue()]}],
        format="json",
        keep_alive=-1,
        options={"temperature": 0, "num_predict": 400},
    )
    return json.loads(resp["message"]["content"])


def clean(s):
    """Lowercase and keep only letters and digits, so 'A)', 'a.' and 'A' all compare equal."""
    return "".join(c for c in s if c.isalnum()).lower()


def _lines_from(img, monitor, scale):
    """OCR one image. Returns lines (with their words) in real screen coordinates."""
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    def box(i):
        left = monitor["left"] + data["left"][i] // scale
        top = monitor["top"] + data["top"][i] // scale
        right = monitor["left"] + (data["left"][i] + data["width"][i]) // scale
        bottom = monitor["top"] + (data["top"][i] + data["height"][i]) // scale
        return left, top, right, bottom

    grouped = {}
    for i, word in enumerate(data["text"]):
        if not word.strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        grouped.setdefault(key, []).append(i)

    lines = []
    for idxs in grouped.values():
        boxes = [box(i) for i in idxs]
        words = [{"text": data["text"][i], "left": b[0], "top": b[1], "right": b[2], "bottom": b[3]}
                 for i, b in zip(idxs, boxes)]
        lines.append({
            "text": " ".join(w["text"] for w in words),
            "left": min(b[0] for b in boxes),
            "top": min(b[1] for b in boxes),
            "right": max(b[2] for b in boxes),
            "bottom": max(b[3] for b in boxes),
            "words": words,
        })
    return lines


def ocr_lines(img, monitor, thorough=False):
    """
    Fast mode: one grayscale pass.
    Thorough mode adds passes for small text, dark text on white (like NEXT),
    and light text on coloured backgrounds (like the answers).
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
    return lines


def center(box):
    return (box["left"] + box["right"]) // 2, (box["top"] + box["bottom"]) // 2


def find_choice_spot(lines, choice, question):
    """
    1. A row that is just the marker (like 'A' or '1'). Click to the right of it, at the tickbox.
    2. Otherwise, a short line containing the choice text. Click the text.
    """
    qc = clean(question)
    label = clean(choice["label"])
    text = clean(choice["text"])

    if label:
        markers = [l for l in lines
                   if clean(l["text"]) == label
                   or (clean(l["text"]).startswith(label)
                       and len(clean(l["text"])) <= len(label) + 1)]
        if markers:
            row = max(markers, key=lambda l: l["top"])
            y = (row["top"] + row["bottom"]) // 2
            return row["right"] + BOX_OFFSET_X, y

    if text:
        hits = [l for l in lines
                if text in clean(l["text"])
                and clean(l["text"]) not in qc
                and len(clean(l["text"])) <= len(text) * 2 + 6]
        if hits:
            return center(min(hits, key=lambda l: len(clean(l["text"]))))

    return None


def find_next_spot(lines, next_label):
    """
    Find the Next button. Checks in order:
      1. A whole line that equals the label.
      2. A single word that equals the label (the NEXT button is often OCR'd
         together with the 'Correct!' header, like 'A Correct! NEXT').
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
    if DRY_RUN:
        print(f"[dry run] would click {desc} at {(x, y)}")
    else:
        pyautogui.click(x, y)
        print(f"Clicked {desc} at {(x, y)}")


def find_chosen(choices, pick):
    p = clean(pick)
    if not p:
        return None
    for c in choices:
        if clean(c["label"]) == p:
            return c
    for c in choices:
        if p == clean(c["text"]) or p in clean(c["text"]):
            return c
    return None


def click_next(next_label):
    """The Next button appears after the tick, so retry a few times."""
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
        if attempt < NEXT_TRIES - 1:
            time.sleep(NEXT_DELAY)

    top = monitor["top"] + monitor["height"] // 3
    seen = [l["text"] for l in last_lines if l["top"] < top and len(l["text"]) <= 25]
    print(f"Couldn't find Next. Short text near the top: {seen}")
    print("Click Next manually.")
    return False


def solve_one():
    t0 = time.perf_counter()
    try:
        img, monitor = grab_screen()
        data = ask_model(img)
        t_model = time.perf_counter()

        question = data.get("question", "")
        choices = data.get("choices", [])
        if not choices:
            print("No answer choices found on screen. Skipping.")
            return

        print(f"\nQ: {question}")
        for c in choices:
            print(f"  {c['label'] or '-'}: {c['text']}")

        chosen = find_chosen(choices, data.get("answer_label", ""))
        if not chosen:
            print(f"Model's pick '{data.get('answer_label', '')}' didn't match any choice. Skipping.")
            return
        print(f"Picked: {chosen['label'] or chosen['text']}")

        lines = ocr_lines(img, monitor)
        spot = find_choice_spot(lines, chosen, question)
        if not spot:
            lines = ocr_lines(img, monitor, thorough=True)
            spot = find_choice_spot(lines, chosen, question)
        if not spot:
            print("Couldn't locate that choice on screen. Skipping.")
            return
        click(*spot, "choice")

        pyautogui.sleep(0.5)
        click_next(data.get("next", "") or "Next")

        t_end = time.perf_counter()
        print(f"Timing: model {t_model - t0:.1f}s | click+next {t_end - t_model:.1f}s | "
              f"total {t_end - t0:.1f}s")
    except Exception as e:
        print(f"Something went wrong: {e}")


def on_hotkey():
    print("Hotkey received, starting...")
    threading.Thread(target=solve_one, daemon=True).start()


if __name__ == "__main__":
    mode = "DRY RUN (no clicks)" if DRY_RUN else "LIVE"
    print(f"Ready in {mode} mode. Click the quiz window, then press Ctrl+Alt+Q.")
    print("Slam the mouse into a screen corner to stop.")
    with keyboard.GlobalHotKeys({"<ctrl>+<alt>+q": on_hotkey}) as listener:
        listener.join()