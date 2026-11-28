# Quiz Bot

A local screen-reading and auto-clicking quiz helper built with Python, Ollama (`qwen2.5vl:3b`), Tesseract OCR, and `pyautogui`.

## Requirements
- Ollama running locally with `qwen2.5vl:3b` pulled (the model itself is unchanged)
- Tesseract OCR installed at `C:\Program Files\Tesseract-OCR\tesseract.exe` (applied on Windows; set `pytesseract.pytesseract.tesseract_cmd` manually on other OSes)
- Python packages: `mss`, `pytesseract`, `pyautogui`, `Pillow`, `pynput`, `ollama`

## Usage
1. Test safely without clicking using `--dry-run` (or set `DRY_RUN = True` in `QUIZ-BOT.py`).
2. Run `python QUIZ-BOT.py`.
3. Focus your quiz window and press `Ctrl+Alt+Q` to solve the current question (presses during an active solve are ignored — no double-clicking).
4. Slam the mouse into any corner of the screen to trigger `pyautogui.FAILSAFE` and stop immediately.

### Offline Bughunt / Replay Mode
Test prompt, OCR, or choice-matching changes against any saved screenshot without clicking or opening a live quiz:
```bash
python QUIZ-BOT.py --test debug_runs/<run_folder>/model_input.png
```
Optional flags:
- `--dry-run` — run the hotkey listener without clicking
- `--no-verify` — skip the second-pass verification on `medium`/`low` confidence answers

## How It Works
1. `[grab]` Screenshot the primary monitor via `mss` (or load `--test` image).
2. `[ocr]` Fast 1-pass Tesseract OCR extracts layout-grouped lines before the model call (`USE_OCR_HINT`), helping the 3B vision model read small text, code, and numbers accurately while reusing the OCR pass for clicking.
3. `[model]` Downscale to `IMG_WIDTH` (`1280px`) and ask `qwen2.5vl:3b` (with `num_ctx=4096`, `num_predict=1024`, `temperature=0`) to output structured JSON:
   - `subject`, `topic`, `context` (headers, passages, code, diagrams, negative/trick wording)
   - `question` and `choices`
   - `reasoning` (step-by-step elimination *before* choosing)
   - `answer_label`, `answer_text`, `confidence`, and `next`
   If `confidence` is `medium`/`low` or `answer_label` and `answer_text` disagree, a second-pass `[verify]` check (`VERIFY_UNCERTAIN`) re-inspects negations, signs, code, and distractors.
4. `[pick]` Cross-checks both `answer_label` and `answer_text` against the extracted choices, recovering from label slips and logging any mismatch.
5. `[locate]` Finds the chosen option on screen (fast OCR first, then 4-pass thorough OCR if needed).
6. `[click]` Clicks the choice, waits `CLICK_SETTLE`, and hunts for the Next button with retries and bottom-strip OCR fallback.

## Bughunting & Logs
- **`quizbot.log`**: Timestamped trace of `[context]`, `[parse]`, `[reason]`, `[verify]`, `[pick]`, `[locate]`, `[next]`, and `[time]`.
- **`debug_runs/run_*/`**: Every solve saves (configurable via `DEBUG_SAVE_MODE` / `DEBUG_KEEP_LAST`):
  - `model_input.png` — exact image sent to `qwen2.5vl:3b`
  - `annotated.png` — screenshot overlaid with yellow OCR boxes and green/blue click crosshairs (`CHOICE` / `NEXT`)
  - `trace.json` — full raw model response, parsed fields, OCR hint, timings, and any error traceback
