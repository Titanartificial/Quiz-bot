# Quiz Bot

A local screen-reading and auto-clicking quiz helper built with Python, Ollama (`qwen2.5vl:3b`), Tesseract OCR, and `pyautogui`.

## Requirements
- [Ollama](https://ollama.com) running locally with the `qwen2.5vl:3b` model pulled
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed at `C:\Program Files\Tesseract-OCR\tesseract.exe` (path is only applied on Windows; set `pytesseract.pytesseract.tesseract_cmd` manually on other OSes)
- Python packages: `mss`, `pytesseract`, `pyautogui`, `Pillow`, `pynput`, `ollama`

## Usage
1. Set `DRY_RUN = True` at the top of `QUIZ-BOT.py` first if you want to test click coordinates safely without clicking
2. Run `QUIZ-BOT.py`
3. Focus your quiz window and press **`Ctrl+Alt+Q`** to solve the current question (presses during an active solve are ignored — no double-clicking)
4. Slam the mouse into any corner of the screen to trigger `pyautogui.FAILSAFE` and stop immediately

## How it works (6 stages, each logged with a `[tag]`)
1. `[grab]` screenshot the primary monitor via `mss`
2. `[model]` downscale + ask `qwen2.5vl:3b` for JSON `{question, choices, answer_label, next}` (same prompt/options as before; robust JSON extraction added)
3. `[pick]` map the model's answer back onto one of its own choices
4. `[locate]` Tesseract OCR finds that choice on screen — fast 1-pass mode, then thorough 4-pass mode (upscale / threshold / inverted-chroma) with duplicate-line dedupe only if needed
5. `[click]` click the option, then poll-retry hunting the Next button (it renders late)
6. `[time]` per-stage timing summary so you can see what is slow

## Debugging
Every run appends timestamped detail to `quizbot.log` (set `LOG_FILE = None` to disable), including a full config dump at startup, the raw model reply snippet when JSON parsing fails, candidate OCR lines when a choice can't be located, and short top-of-screen text when the Next button can't be found. The module docstring in `QUIZ-BOT.py` contains a cheat-sheet mapping each log tag to its likely cause and the CONFIG knob to tweak (`BOX_OFFSET_X`, `NEXT_LABELS`, `CHOICE_LEN_FACTOR/SLACK`, `MAX_MARKER_EXTRA_CHARS`, `NEXT_TRIES/DELAY`).

## Robustness notes
- Model output wrapped in ``` fences or stray prose no longer crashes the run
- Zero-confidence OCR noise tokens are dropped before matching
- Duplicate detections from overlapping OCR passes are removed
- Overlapping hotkey presses are serialized with a lock
- Exceptions inside a solve cycle are logged with traceback instead of killing the worker silently
