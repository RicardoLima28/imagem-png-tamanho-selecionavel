"""
Background Remover — Desktop GUI Application
============================================

A desktop tool that lets the user:
  * Pick one or many images from disk.
  * Automatically remove the background using AI (the `rembg` library).
  * Resize the result to a chosen width/height (optionally keeping aspect ratio).
  * Preview the transparent result before saving.
  * Save the output as a transparent PNG into a chosen folder.

Stack:
  * GUI .......... CustomTkinter (themed Tkinter wrapper, supports dark mode).
  * AI removal ... rembg (ONNX based U^2-Net models).
  * Imaging ...... Pillow (resizing, compositing, PNG export).
  * Drag & drop .. tkinterdnd2 (optional; the app still works without it).

All long-running work (the AI inference) runs on a background thread so the
interface never freezes; progress and status are pushed back to the GUI thread
through a thread-safe queue.

Run with:  python main.py
"""

from __future__ import annotations

import io
import os
import queue
import threading
import traceback
from dataclasses import dataclass, field
from typing import Callable, Optional

import customtkinter as ctk
import numpy as np
from PIL import Image
from tkinter import filedialog, messagebox

# ---------------------------------------------------------------------------
# Optional drag-and-drop support.
# tkinterdnd2 is optional: if it is not installed the program still runs, it
# simply will not accept files dropped onto the window. We detect it lazily so
# a missing dependency never crashes the app.
# ---------------------------------------------------------------------------
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    _DND_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the environment
    DND_FILES = None
    TkinterDnD = None
    _DND_AVAILABLE = False


# Image extensions we are willing to open as input.
SUPPORTED_INPUT_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif")


# ===========================================================================
# Core image processing (pure logic, no GUI) — easy to test in isolation.
# ===========================================================================
@dataclass
class ProcessOptions:
    """Bundle of user-selected options that control how an image is processed."""

    width: Optional[int] = None          # Target width in pixels (None = keep original).
    height: Optional[int] = None         # Target height in pixels (None = keep original).
    keep_aspect_ratio: bool = True       # If True, fit inside (width, height) without distortion.
    output_dir: str = ""                 # Folder where PNGs are written.


def remove_background(image: Image.Image, session=None) -> Image.Image:
    """Run AI background removal on a Pillow image and return an RGBA image.

    `rembg` is imported lazily here (not at module import time) because loading
    it pulls in onnxruntime and downloads the model on first use, which is slow.
    Importing it only when needed keeps app startup fast.
    """
    from rembg import remove  # Lazy import: heavy dependency.

    # rembg.remove accepts a PIL image and returns a PIL image with an alpha
    # channel where the background pixels have been made transparent.
    result = remove(image, session=session)
    return result.convert("RGBA")


def resize_image(image: Image.Image, opts: ProcessOptions) -> Image.Image:
    """Resize `image` according to the options.

    Three behaviours:
      * No width and no height  -> return the image unchanged.
      * keep_aspect_ratio=True  -> scale to fit *inside* the given box, never
        distorting the picture (the smaller of the two scale factors wins).
      * keep_aspect_ratio=False -> stretch exactly to (width, height).
    """
    w, h = opts.width, opts.height

    # Nothing requested: leave the image as-is.
    if not w and not h:
        return image

    orig_w, orig_h = image.size

    if opts.keep_aspect_ratio:
        # Determine the limiting dimension. If only one side was given, derive
        # the other from the original aspect ratio.
        if w and h:
            scale = min(w / orig_w, h / orig_h)
        elif w:
            scale = w / orig_w
        else:  # only height given
            scale = h / orig_h
        new_size = (max(1, round(orig_w * scale)), max(1, round(orig_h * scale)))
    else:
        # Stretch to fill. Fall back to the original side if one was left blank.
        new_size = (w or orig_w, h or orig_h)

    # Nothing to do if the size did not actually change.
    if new_size == (orig_w, orig_h):
        return image

    # Imagens com transparência (RGBA) precisam ser redimensionadas com o alfa
    # pré-multiplicado, senão as bordas semitransparentes "puxam" a cor do
    # fundo antigo e aparece uma franja/halo — o que parece perda de qualidade.
    if image.mode == "RGBA":
        return _resize_rgba_premultiplied(image, new_size)

    # LANCZOS gives the best quality for downscaling/upscaling photos.
    return image.resize(new_size, Image.LANCZOS)


def _resize_rgba_premultiplied(image: Image.Image, new_size: tuple[int, int]) -> Image.Image:
    """Redimensiona uma imagem RGBA usando alfa pré-multiplicado.

    Sem isso, o Pillow interpola os canais de cor de forma independente do
    alfa: pixels totalmente transparentes (que ainda guardam a cor original do
    fundo) sangram para dentro das bordas do recorte, criando um halo. Ao
    multiplicar a cor pelo alfa antes de redimensionar e dividir de volta
    depois, as bordas ficam limpas e nítidas.
    """
    arr = np.asarray(image, dtype=np.float32)
    rgb = arr[..., :3]
    alpha = arr[..., 3:4] / 255.0

    # Pré-multiplica a cor pelo alfa e redimensiona cor + alfa juntos.
    premult = np.concatenate([rgb * alpha, arr[..., 3:4]], axis=-1)
    premult_img = Image.fromarray(np.clip(premult, 0, 255).astype(np.uint8), "RGBA")
    resized = premult_img.resize(new_size, Image.LANCZOS)

    # Desfaz a pré-multiplicação para voltar à cor real (evitando divisão por 0).
    out = np.asarray(resized, dtype=np.float32)
    out_alpha = out[..., 3:4] / 255.0
    rgb_back = np.divide(
        out[..., :3], out_alpha,
        out=np.zeros_like(out[..., :3]), where=out_alpha > 0,
    )
    final = np.concatenate([np.clip(rgb_back, 0, 255), out[..., 3:4]], axis=-1)
    return Image.fromarray(final.astype(np.uint8), "RGBA")


def process_one(input_path: str, opts: ProcessOptions, session=None) -> Image.Image:
    """Full pipeline for a single file: open -> remove bg -> resize.

    Returns the finished RGBA image (it is the caller's job to save it).
    """
    with Image.open(input_path) as img:
        # Convert up front so modes like "P" (palette) or "L" (grayscale) work.
        img = img.convert("RGBA")
        cutout = remove_background(img, session=session)
        cutout = resize_image(cutout, opts)
    return cutout


def output_path_for(input_path: str, output_dir: str) -> str:
    """Build the destination path: <output_dir>/<original name>.png."""
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(output_dir, f"{base}.png")


# ===========================================================================
# Worker thread + message queue.
# ===========================================================================
@dataclass
class WorkerMessage:
    """A message sent from the worker thread back to the GUI thread."""

    kind: str                            # "progress" | "preview" | "done" | "error"
    payload: object = None
    progress: float = 0.0                # 0.0 .. 1.0
    text: str = ""


class ProcessingWorker(threading.Thread):
    """Processes a list of images on a background thread.

    Communicates exclusively through a thread-safe `queue.Queue`; it never
    touches Tkinter widgets directly (that is illegal from a non-GUI thread).
    """

    def __init__(self, files: list[str], opts: ProcessOptions, out_queue: "queue.Queue[WorkerMessage]"):
        super().__init__(daemon=True)
        self.files = files
        self.opts = opts
        self.queue = out_queue
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Request cooperative cancellation; checked between files."""
        self._cancel.set()

    def run(self) -> None:  # Executed on the background thread.
        try:
            # Create a single rembg session and reuse it for every file so the
            # model is only loaded once (big speed-up for batch jobs).
            from rembg import new_session

            self.queue.put(WorkerMessage("progress", progress=0.02,
                                         text="Carregando modelo de IA…"))
            session = new_session("u2net")

            total = len(self.files)
            last_image: Optional[Image.Image] = None

            for index, path in enumerate(self.files):
                if self._cancel.is_set():
                    self.queue.put(WorkerMessage("error", text="Cancelado pelo usuário."))
                    return

                name = os.path.basename(path)
                self.queue.put(WorkerMessage(
                    "progress",
                    progress=(index) / total,
                    text=f"Processando {name} ({index + 1}/{total})…",
                ))

                # Heavy lifting for this single file.
                result = process_one(path, self.opts, session=session)
                last_image = result

                # Save as a transparent PNG. PNG always preserves the alpha
                # channel, satisfying the "transparent PNG only" requirement.
                dest = output_path_for(path, self.opts.output_dir)
                result.save(dest, "PNG")

                # Push a preview of the most recent result to the GUI.
                self.queue.put(WorkerMessage("preview", payload=result.copy()))

            self.queue.put(WorkerMessage(
                "done",
                progress=1.0,
                payload=last_image.copy() if last_image else None,
                text=f"Concluído — {total} imagem(ns) salva(s) em:\n{self.opts.output_dir}",
            ))
        except Exception:  # Any failure is reported to the GUI, never silently swallowed.
            self.queue.put(WorkerMessage("error", text=traceback.format_exc()))


# ===========================================================================
# Helper: render an RGBA image onto a checkerboard so transparency is visible.
# ===========================================================================
def make_checkerboard(size: tuple[int, int], square: int = 12) -> Image.Image:
    """Create a gray/white checkerboard image used as a preview backdrop."""
    w, h = size
    board = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    light = (255, 255, 255, 255)
    dark = (205, 205, 205, 255)
    px = board.load()
    for y in range(h):
        for x in range(w):
            # Alternate colour every `square` pixels in both directions.
            if (x // square + y // square) % 2 == 0:
                px[x, y] = light
            else:
                px[x, y] = dark
    return board


def composite_for_preview(image: Image.Image, max_side: int = 360) -> Image.Image:
    """Shrink `image` to fit `max_side` and lay it over a checkerboard.

    The checkerboard makes the transparent regions obvious to the user.
    """
    img = image.copy()
    # Calcula o tamanho da miniatura preservando a proporção, e usa o mesmo
    # redimensionamento com alfa pré-multiplicado para a prévia não ganhar halo.
    w, h = img.size
    scale = min(max_side / w, max_side / h, 1.0)
    if scale < 1.0:
        thumb_size = (max(1, round(w * scale)), max(1, round(h * scale)))
        img = _resize_rgba_premultiplied(img, thumb_size) if img.mode == "RGBA" \
            else img.resize(thumb_size, Image.LANCZOS)
    board = make_checkerboard(img.size)
    board.alpha_composite(img)
    return board


# ===========================================================================
# Main application window.
# ===========================================================================
# Pick the base class: TkinterDnD.Tk when drag-and-drop is available, otherwise
# CustomTkinter's CTk. We build a small subclass so the rest of the code does
# not care which one is in use.
_BaseWindow = TkinterDnD.Tk if _DND_AVAILABLE else ctk.CTk


class BackgroundRemoverApp(_BaseWindow):
    def __init__(self):
        super().__init__()

        # When we inherit from TkinterDnD.Tk (a plain Tk root) CustomTkinter's
        # automatic theming of the root window does not run, so apply it manually.
        if _DND_AVAILABLE:
            ctk.set_appearance_mode("dark")
            # CTk normally configures the root background; do it ourselves.
            self.configure(bg="#242424")

        # ---- Window basics -------------------------------------------------
        self.title("Removedor de Fundo com IA")
        self.geometry("900x600")
        self.minsize(820, 560)

        # ---- State ---------------------------------------------------------
        self.selected_files: list[str] = []     # Input image paths.
        self.output_dir: str = ""                # Destination folder.
        self.worker: Optional[ProcessingWorker] = None
        self.msg_queue: "queue.Queue[WorkerMessage]" = queue.Queue()
        self._preview_imgtk = None               # Keep a reference so it isn't GC'd.

        # ---- Appearance defaults (dark mode on by default) -----------------
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Build the two-column layout.
        self._build_layout()

        # Start the periodic queue poller that bridges worker -> GUI.
        self.after(100, self._poll_queue)

        # Wire up drag-and-drop if the library is present.
        if _DND_AVAILABLE:
            self._enable_dnd()

    # ------------------------------------------------------------------ UI --
    def _build_layout(self) -> None:
        """Create all widgets. Left column = controls, right column = preview."""
        self.grid_columnconfigure(0, weight=0)   # Controls: fixed width.
        self.grid_columnconfigure(1, weight=1)   # Preview: expands.
        self.grid_rowconfigure(0, weight=1)

        # ===================== LEFT: control panel ==========================
        controls = ctk.CTkScrollableFrame(self, width=320, corner_radius=0)
        controls.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(
            controls, text="Removedor de Fundo com IA",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(padx=20, pady=(20, 4), anchor="w")

        ctk.CTkLabel(
            controls,
            text="Remova fundos e exporte PNGs transparentes.",
            text_color=("gray40", "gray70"), wraplength=280, justify="left",
        ).pack(padx=20, pady=(0, 16), anchor="w")

        # --- 1) Image selection ----------------------------------------
        ctk.CTkButton(controls, text="📂  Selecionar Imagem(ns)…",
                      command=self._on_select_images).pack(padx=20, pady=(4, 4), fill="x")

        self.files_label = ctk.CTkLabel(
            controls, text="Nenhuma imagem selecionada.",
            text_color=("gray40", "gray70"), wraplength=280, justify="left",
        )
        self.files_label.pack(padx=20, pady=(0, 12), anchor="w")

        # --- 6) Output folder ------------------------------------------
        ctk.CTkButton(controls, text="📁  Selecionar Pasta de Saída…",
                      command=self._on_select_output).pack(padx=20, pady=(4, 4), fill="x")

        self.output_label = ctk.CTkLabel(
            controls, text="Nenhuma pasta de saída selecionada.",
            text_color=("gray40", "gray70"), wraplength=280, justify="left",
        )
        self.output_label.pack(padx=20, pady=(0, 12), anchor="w")

        # --- 4) Output resolution --------------------------------------
        ctk.CTkLabel(controls, text="Resolução de saída (pixels)",
                     font=ctk.CTkFont(weight="bold")).pack(padx=20, pady=(8, 2), anchor="w")

        res_frame = ctk.CTkFrame(controls, fg_color="transparent")
        res_frame.pack(padx=20, pady=(0, 4), fill="x")
        res_frame.grid_columnconfigure((0, 1), weight=1)

        # Width / height entries. Left blank = "keep original on this axis".
        self.width_entry = ctk.CTkEntry(res_frame, placeholder_text="Largura")
        self.width_entry.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self.height_entry = ctk.CTkEntry(res_frame, placeholder_text="Altura")
        self.height_entry.grid(row=0, column=1, padx=(4, 0), sticky="ew")

        # --- 5/aspect) Preserve aspect ratio ---------------------------
        self.keep_aspect_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(controls, text="Preservar proporção",
                        variable=self.keep_aspect_var).pack(padx=20, pady=(8, 4), anchor="w")

        # Quick preset buttons for common sizes.
        preset_frame = ctk.CTkFrame(controls, fg_color="transparent")
        preset_frame.pack(padx=20, pady=(0, 12), fill="x")
        for i, (label, size) in enumerate(
            [("512²", (512, 512)), ("1024²", (1024, 1024)), ("Original", (None, None))]  # "Original" já está em português
        ):
            ctk.CTkButton(
                preset_frame, text=label, width=60, height=26,
                fg_color=("gray75", "gray30"), hover_color=("gray65", "gray40"),
                command=lambda s=size: self._apply_preset(s),
            ).pack(side="left", padx=(0 if i == 0 else 6, 0))

        # --- Appearance toggle (dark mode requirement) -----------------
        ctk.CTkLabel(controls, text="Aparência",
                     font=ctk.CTkFont(weight="bold")).pack(padx=20, pady=(8, 2), anchor="w")
        self.appearance_menu = ctk.CTkOptionMenu(
            controls, values=["Escuro", "Claro", "Sistema"],
            command=self._on_appearance_change,
        )
        self.appearance_menu.set("Escuro")
        self.appearance_menu.pack(padx=20, pady=(0, 16), fill="x")

        # --- 2/3) Process button ---------------------------------------
        self.process_btn = ctk.CTkButton(
            controls, text="✨  Remover Fundo", height=42,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self._on_process,
        )
        self.process_btn.pack(padx=20, pady=(4, 8), fill="x")

        self.cancel_btn = ctk.CTkButton(
            controls, text="Cancelar", height=32, state="disabled",
            fg_color=("gray70", "gray35"), hover_color=("gray60", "gray45"),
            command=self._on_cancel,
        )
        self.cancel_btn.pack(padx=20, pady=(0, 16), fill="x")

        # ===================== RIGHT: preview panel =========================
        preview_panel = ctk.CTkFrame(self, corner_radius=0)
        preview_panel.grid(row=0, column=1, sticky="nsew")
        preview_panel.grid_rowconfigure(0, weight=1)
        preview_panel.grid_columnconfigure(0, weight=1)

        # 7) Preview area. We show text instructions until an image is ready.
        self.preview_label = ctk.CTkLabel(
            preview_panel,
            text=("Arraste uma imagem aqui\n\n(ou use “Selecionar Imagem(ns)”)"
                  if _DND_AVAILABLE else "A prévia aparecerá aqui"),
            font=ctk.CTkFont(size=15), text_color=("gray45", "gray60"),
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

        # 8) Progress bar + 9) status message live in a bottom bar.
        bottom = ctk.CTkFrame(preview_panel, fg_color="transparent")
        bottom.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 20))
        bottom.grid_columnconfigure(0, weight=1)

        self.progress = ctk.CTkProgressBar(bottom)
        self.progress.set(0)
        self.progress.grid(row=0, column=0, sticky="ew")

        self.status_label = ctk.CTkLabel(
            bottom, text="Pronto.", anchor="w",
            text_color=("gray35", "gray70"), wraplength=480, justify="left",
        )
        self.status_label.grid(row=1, column=0, sticky="ew", pady=(8, 0))

    # ---------------------------------------------------------- DnD setup --
    def _enable_dnd(self) -> None:
        """Register the preview area (and window) as a drop target for files."""
        for widget in (self, self.preview_label):
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind("<<Drop>>", self._on_drop)

    def _on_drop(self, event):
        """Handle files dropped onto the window."""
        # event.data is a brace/space separated list of paths from the OS.
        raw = self.tk.splitlist(event.data)
        files = [p for p in raw if p.lower().endswith(SUPPORTED_INPUT_EXTS)]
        if files:
            self._set_files(files)
        else:
            messagebox.showwarning("Arquivo não suportado",
                                   "Por favor, solte arquivos de imagem (PNG, JPG, WEBP, …).")

    # ------------------------------------------------------- UI callbacks --
    def _apply_preset(self, size: tuple[Optional[int], Optional[int]]) -> None:
        """Fill the width/height entries from a preset button."""
        w, h = size
        self.width_entry.delete(0, "end")
        self.height_entry.delete(0, "end")
        if w:
            self.width_entry.insert(0, str(w))
        if h:
            self.height_entry.insert(0, str(h))

    def _on_appearance_change(self, choice: str) -> None:
        """Switch between Dark / Light / System appearance modes."""
        # Mapeia os rótulos em português para os modos que o CustomTkinter espera.
        modes = {"Escuro": "dark", "Claro": "light", "Sistema": "system"}
        ctk.set_appearance_mode(modes.get(choice, "dark"))

    def _on_select_images(self) -> None:
        """Open a file dialog allowing one or many images (batch processing)."""
        paths = filedialog.askopenfilenames(
            title="Selecionar imagem(ns)",
            filetypes=[("Imagens", " ".join(f"*{e}" for e in SUPPORTED_INPUT_EXTS)),
                       ("Todos os arquivos", "*.*")],
        )
        if paths:
            self._set_files(list(paths))

    def _set_files(self, files: list[str]) -> None:
        """Store the chosen input files and update the label."""
        self.selected_files = files
        if len(files) == 1:
            self.files_label.configure(text=os.path.basename(files[0]))
        else:
            self.files_label.configure(text=f"{len(files)} imagens selecionadas (lote).")
        self._set_status(f"{len(files)} arquivo(s) pronto(s).")

    def _on_select_output(self) -> None:
        """Choose the destination folder for the exported PNGs."""
        folder = filedialog.askdirectory(title="Selecionar pasta de saída")
        if folder:
            self.output_dir = folder
            self.output_label.configure(text=folder)

    def _read_dimensions(self) -> tuple[Optional[int], Optional[int]]:
        """Parse and validate the width/height entries.

        Returns (width, height) where either may be None (blank = keep original).
        Raises ValueError with a friendly message on bad input.
        """
        def parse(entry: ctk.CTkEntry, name: str) -> Optional[int]:
            text = entry.get().strip()
            if not text:
                return None
            if not text.isdigit() or int(text) <= 0:
                raise ValueError(f"{name} deve ser um número inteiro positivo.")
            return int(text)

        return parse(self.width_entry, "A largura"), parse(self.height_entry, "A altura")

    # ----------------------------------------------------- Processing flow --
    def _on_process(self) -> None:
        """Validate inputs and kick off the background worker thread."""
        # --- Validation (requirement 9: clear error messages) -------------
        if not self.selected_files:
            messagebox.showerror("Nenhuma imagem", "Por favor, selecione ao menos uma imagem primeiro.")
            return
        if not self.output_dir:
            messagebox.showerror("Nenhuma pasta de saída", "Por favor, escolha uma pasta de saída.")
            return

        try:
            width, height = self._read_dimensions()
        except ValueError as exc:
            messagebox.showerror("Resolução inválida", str(exc))
            return

        opts = ProcessOptions(
            width=width, height=height,
            keep_aspect_ratio=self.keep_aspect_var.get(),
            output_dir=self.output_dir,
        )

        # --- Lock the UI while working -----------------------------------
        self._set_busy(True)
        self.progress.set(0)
        self._set_status("Iniciando…")

        # --- Launch the worker -------------------------------------------
        self.worker = ProcessingWorker(self.selected_files, opts, self.msg_queue)
        self.worker.start()

    def _on_cancel(self) -> None:
        """Ask the running worker to stop after its current file."""
        if self.worker and self.worker.is_alive():
            self.worker.cancel()
            self._set_status("Cancelando…")

    # ---------------------------------------------- worker -> GUI bridge --
    def _poll_queue(self) -> None:
        """Runs on the GUI thread ~10x/sec; drains messages from the worker.

        This is the *only* place worker results touch Tkinter widgets, which
        keeps all GUI mutation on the main thread (a Tk requirement).
        """
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                self._handle_message(msg)
        except queue.Empty:
            pass
        # Reschedule.
        self.after(100, self._poll_queue)

    def _handle_message(self, msg: WorkerMessage) -> None:
        if msg.kind == "progress":
            self.progress.set(msg.progress)
            if msg.text:
                self._set_status(msg.text)

        elif msg.kind == "preview":
            self._show_preview(msg.payload)

        elif msg.kind == "done":
            self.progress.set(1.0)
            if msg.payload is not None:
                self._show_preview(msg.payload)
            self._set_busy(False)
            self._set_status(msg.text)
            messagebox.showinfo("Sucesso", msg.text)  # Requirement 9: success message.

        elif msg.kind == "error":
            self._set_busy(False)
            self.progress.set(0)
            self._set_status("Erro.")
            messagebox.showerror("Falha no processamento", msg.text)  # Requirement 9.

    # ------------------------------------------------------------ helpers --
    def _show_preview(self, image: Optional[Image.Image]) -> None:
        """Display an RGBA result in the preview pane over a checkerboard."""
        if image is None:
            return
        composed = composite_for_preview(image, max_side=380)
        # CTkImage handles HiDPI scaling and keeps the image crisp.
        self._preview_imgtk = ctk.CTkImage(
            light_image=composed, dark_image=composed, size=composed.size,
        )
        self.preview_label.configure(image=self._preview_imgtk, text="")

    def _set_status(self, text: str) -> None:
        self.status_label.configure(text=text)

    def _set_busy(self, busy: bool) -> None:
        """Enable/disable controls while a job is running."""
        self.process_btn.configure(state="disabled" if busy else "normal")
        self.cancel_btn.configure(state="normal" if busy else "disabled")


# ===========================================================================
# Entry point.
# ===========================================================================
def main() -> None:
    app = BackgroundRemoverApp()
    app.mainloop()


if __name__ == "__main__":
    main()
