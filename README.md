# AI Background Remover

A Python desktop application that removes image backgrounds with AI and exports
transparent PNGs at a resolution you choose.

![interface](https://img.shields.io/badge/GUI-CustomTkinter-blue) ![ai](https://img.shields.io/badge/AI-rembg-green)

## Features

| # | Requirement | Status |
|---|-------------|--------|
| 1 | Select an image from disk | ✅ file dialog |
| 2 | Automatic AI background removal | ✅ `rembg` (U²-Net) |
| 3 | Save as transparent PNG | ✅ always RGBA PNG |
| 4 | Choose output width / height | ✅ entry fields + presets |
| 5 | Resize to chosen resolution | ✅ Pillow LANCZOS |
| 6 | Select an output folder | ✅ folder dialog |
| 7 | Preview the result before saving | ✅ checkerboard preview |
| 8 | Progress bar during processing | ✅ live progress |
| 9 | Success / error messages | ✅ dialogs + status bar |
| 10 | CustomTkinter + rembg | ✅ |

**Optional extras included:**
- 🗂️ **Batch processing** — select many images at once.
- 🖱️ **Drag & drop** — drop files onto the window (needs `tkinterdnd2`).
- 🔒 **Transparent-PNG-only** export.
- 📐 **Preserve aspect ratio** toggle.
- 🌗 **Dark mode** (default) with Light / System switch.

## Installation

> Requires **Python 3.9–3.12** recommended. `rembg`/`onnxruntime` wheels may not
> yet exist for the very newest Python releases — if `pip install` fails on
> `onnxruntime`, use a supported Python version (e.g. 3.11 / 3.12).

```powershell
# 1. (recommended) create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. install dependencies
pip install -r requirements.txt
```

The first time you remove a background, `rembg` downloads its model
(~170 MB) automatically; this happens once.

## Usage

```powershell
python main.py
```

1. Click **Select Image(s)** (or drag images onto the window).
2. Click **Select Output Folder**.
3. By default the app **only removes the background** and keeps the original
   size. To resize, turn on **Redimensionar imagem**, then type a **Width** /
   **Height** or use a preset, and toggle **Preserve aspect ratio** as needed.
4. Click **Remove Background**. Watch the progress bar; the result previews on
   the right over a checkerboard (so you can see the transparency).
5. Finished PNGs are written to your output folder using the original filenames.

## How it works

- **`main.py`** is split into a pure *processing core* (open → remove background
  → resize → save) and the *GUI*. The core functions (`remove_background`,
  `resize_image`, `process_one`) have no Tkinter dependency and are easy to test.
- All AI work runs on a **background thread** (`ProcessingWorker`) so the window
  never freezes. The worker talks to the GUI only through a thread-safe
  `queue.Queue`, and the GUI drains that queue ~10×/second — keeping every
  widget update on the main thread as Tkinter requires.
- A single `rembg` session is reused for the whole batch, so the model loads
  only once.

## Notes & troubleshooting

- **Drag & drop not working?** `tkinterdnd2` may not be installed. The rest of
  the app still works; install it with `pip install tkinterdnd2`.
- **First run is slow** — that's the one-time model download.
- **`onnxruntime` install error** — your Python version may be too new; use a
  supported version or `pip install onnxruntime` separately to see the error.
