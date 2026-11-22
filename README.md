# Quiz Bot

A local screen-reading and auto-clicking quiz helper built with Python, Ollama (`qwen2.5vl:3b`), Tesseract OCR, and `pyautogui`[cite: 1].

## Requirements
- [Ollama](https://ollama.com) running locally with the `qwen2.5vl:3b` model pulled[cite: 1]
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed at `C:\Program Files\Tesseract-OCR\tesseract.exe`[cite: 1]
- Python packages: `mss`, `pytesseract`, `pyautogui`, `Pillow`, `pynput`, `ollama`[cite: 1]

## Usage
1. Set `DRY_RUN = True` at the top of `QUIZ-BOT.py` first if you want to test click coordinates safely without clicking[cite: 1].
2. Run `QUIZ-BOT.py`[cite: 1].
3. Focus your quiz window and press **`Ctrl+Alt+Q`** to solve the current question[cite: 1].
4. Slam the mouse into any corner of the screen to trigger `pyautogui.FAILSAFE` and stop immediately[cite: 1].
