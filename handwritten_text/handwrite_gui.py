"""
Tkinter front-end for the handwriting pipeline.

Type Markdown+LaTeX / plain LaTeX / plain text directly, or import a
.md/.tex/.txt file, tune the same options the CLIs expose, and generate a
lazily-streamed PNG-per-page + PDF using the current hw/ engine
(iter_document, --hand-math, ink texture, paper scan, etc.).

Keeps the good bones of this project's original Tkinter app (app.py): a
background thread so the window never freezes, a progress callback, and a
save-file dialog -- but wired to the new engine instead of the old cv2
glyph-bank renderer app.py depended on.

  python handwrite_gui.py
"""
import os
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk

from hw.render import HandwritingRenderer
from hw import markdown_render, latex_render

MODES = ("Markdown + LaTeX", "LaTeX (documento completo)", "Texto simples")

DEMO_TEXT = (
    "# Minha nota\n\n"
    "Qualquer texto aqui, inclusive matematica: $E[X] = \\mu$ e "
    "$\\int_0^1 x^2\\,dx = \\frac{1}{3}$.\n"
)


def blocks_for_mode(mode, text):
    """Turn raw editor text into the block list HandwritingRenderer expects,
    using whichever parser matches the selected content mode."""
    if mode == MODES[0]:
        return markdown_render.parse(text)
    if mode == MODES[1]:
        return latex_render.parse(text)
    return [{"type": "para", "gap": 0.15,
            "runs": [("t", ln)] if ln.strip() else [("t", " ")]}
           for ln in text.split("\n")]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Escrita a Mao - gerador")
        self.geometry("1000x780")
        self.minsize(780, 560)
        self._preview_imgtk = None   # keep a reference so Tk doesn't GC it
        self._busy = False
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- layout -------------------------------------------------------
    def _build(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="Conteudo:").pack(side="left")
        self.mode_var = tk.StringVar(value=MODES[0])
        ttk.Combobox(top, textvariable=self.mode_var, values=MODES,
                    state="readonly", width=26).pack(side="left", padx=6)
        ttk.Button(top, text="Importar arquivo...",
                  command=self._import_file).pack(side="left", padx=6)
        ttk.Button(top, text="Limpar", command=self._clear_text).pack(side="left")

        self.text = tk.Text(self, wrap="word", undo=True, font=("Consolas", 11))
        self.text.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.text.insert("1.0", DEMO_TEXT)

        opts = ttk.LabelFrame(self, text="Opcoes", padding=8)
        opts.pack(fill="x", padx=8)

        self.hand_math_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Matematica na minha letra (--hand-math)",
                        variable=self.hand_math_var).grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.scan_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Papel escaneado (--scan)",
                        variable=self.scan_var).grid(row=0, column=1, sticky="w", padx=4, pady=2)
        self.ruled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Linhas de caderno (--ruled)",
                        variable=self.ruled_var).grid(row=0, column=2, sticky="w", padx=4, pady=2)
        self.model_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Rede neural p/ letras faltantes (--model)",
                        variable=self.model_var).grid(row=0, column=3, sticky="w", padx=4, pady=2)

        ttk.Label(opts, text="Intensidade do papel:").grid(row=1, column=0, sticky="w", padx=4, pady=(4, 0))
        self.scan_strength = tk.DoubleVar(value=1.0)
        ttk.Spinbox(opts, from_=0.2, to=2.0, increment=0.1, width=6,
                   textvariable=self.scan_strength).grid(row=1, column=1, sticky="w", pady=(4, 0))
        ttk.Label(opts, text="Semente (vazio = aleatoria):").grid(row=1, column=2, sticky="w", padx=4, pady=(4, 0))
        self.seed_var = tk.StringVar(value="")
        ttk.Entry(opts, width=8, textvariable=self.seed_var).grid(row=1, column=3, sticky="w", pady=(4, 0))

        adv = ttk.LabelFrame(self, text="Avancado", padding=8)
        adv.pack(fill="x", padx=8, pady=(6, 0))

        def labeled_spin(parent, label, var, lo, hi, step, row, col):
            ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(4, 2), pady=2)
            ttk.Spinbox(parent, from_=lo, to=hi, increment=step, width=6,
                       textvariable=var).grid(row=row, column=col + 1, sticky="w", pady=2)

        self.ink_var = tk.DoubleVar(value=0.8)
        self.tremor_var = tk.DoubleVar(value=0.3)
        self.regularize_var = tk.DoubleVar(value=1.0)
        self.stroke_var = tk.DoubleVar(value=0.11)
        self.math_style_var = tk.DoubleVar(value=1.0)
        self.xh_var = tk.IntVar(value=26)
        self.width_var = tk.IntVar(value=1000)

        labeled_spin(adv, "Tinta:", self.ink_var, 0.0, 1.5, 0.1, 0, 0)
        labeled_spin(adv, "Tremor:", self.tremor_var, 0.0, 1.5, 0.1, 0, 2)
        labeled_spin(adv, "Regularizar:", self.regularize_var, 0.0, 1.0, 0.1, 0, 4)
        labeled_spin(adv, "Estilo da matematica:", self.math_style_var, 0.0, 2.0, 0.1, 1, 0)
        labeled_spin(adv, "Tamanho da letra (xh):", self.xh_var, 14, 60, 1, 1, 2)
        labeled_spin(adv, "Largura da pagina (px):", self.width_var, 500, 2000, 50, 1, 4)

        bottom = ttk.Frame(self, padding=8)
        bottom.pack(fill="x")
        self.generate_btn = ttk.Button(bottom, text="Gerar", command=self._start_generation)
        self.generate_btn.pack(side="left")
        self.progress = ttk.Progressbar(bottom, mode="determinate", length=300)
        self.progress.pack(side="left", padx=10)
        self.status_var = tk.StringVar(value="Pronto.")
        ttk.Label(bottom, textvariable=self.status_var).pack(side="left", padx=6)

        self.preview_label = ttk.Label(self)
        self.preview_label.pack(padx=8, pady=(0, 8))

    # ---- actions --------------------------------------------------------
    def _clear_text(self):
        self.text.delete("1.0", "end")

    def _import_file(self):
        path = filedialog.askopenfilename(
            title="Importar documento",
            filetypes=[("Documentos", "*.md *.tex *.txt"), ("Todos os arquivos", "*.*")])
        if not path:
            return
        with open(path, encoding="utf-8", errors="replace") as f:
            content = f.read()
        self.text.delete("1.0", "end")
        self.text.insert("1.0", content)
        ext = os.path.splitext(path)[1].lower()
        self.mode_var.set(MODES[0] if ext == ".md" else
                          MODES[1] if ext == ".tex" else MODES[2])

    def _start_generation(self):
        if self._busy:
            return
        text = self.text.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Aviso", "Digite ou importe algum conteudo primeiro.")
            return

        out_path = filedialog.asksaveasfilename(
            title="Salvar como", defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if not out_path:
            return
        out_base = out_path[:-4] if out_path.lower().endswith(".pdf") else out_path

        seed_txt = self.seed_var.get().strip()
        try:
            seed = int(seed_txt) if seed_txt else None
        except ValueError:
            messagebox.showerror("Erro", "Semente precisa ser um numero inteiro (ou vazio).")
            return

        try:
            blocks = blocks_for_mode(self.mode_var.get(), text)
        except Exception as exc:
            messagebox.showerror("Erro ao interpretar o conteudo", str(exc))
            return
        if not blocks:
            messagebox.showwarning("Aviso", "Nao ha conteudo reconhecivel para renderizar.")
            return

        opts = dict(
            hand_math=self.hand_math_var.get(), math_style=self.math_style_var.get(),
            regularize=self.regularize_var.get(), stroke_ratio=self.stroke_var.get(),
            ink_texture=self.ink_var.get(), letter_tremor=self.tremor_var.get(),
            model=self.model_var.get(), scan=self.scan_var.get(),
            scan_strength=self.scan_strength.get(), ruled=self.ruled_var.get(),
            xh=self.xh_var.get(), width=self.width_var.get(), seed=seed,
        )

        self._busy = True
        self.generate_btn.config(state="disabled")
        self.progress.config(mode="determinate", value=0, maximum=max(1, len(blocks)))
        self.status_var.set("Iniciando...")

        threading.Thread(target=self._run, args=(blocks, out_base, opts), daemon=True).start()

    def _run(self, blocks, out_base, opts):
        """Runs in a background thread -- never touch widgets directly here,
        only via self.after(...) so Tk stays on the main thread."""
        t0 = time.time()
        try:
            model = None
            if opts["model"]:
                ck = "hw/checkpoints/best.pt"
                if os.path.exists(ck):
                    from hw.model import GlyphGenerator
                    model = GlyphGenerator(ck)

            renderer = HandwritingRenderer(
                seed=opts["seed"], model=model, hand_math=opts["hand_math"],
                math_style=opts["math_style"], regularize=opts["regularize"],
                stroke_ratio=opts["stroke_ratio"], ink_texture=opts["ink_texture"],
                letter_tremor=opts["letter_tremor"])

            scan_fn = None
            if opts["scan"]:
                from hw.paper import scan_effect
                base_seed = opts["seed"] if opts["seed"] is not None else 0
                scan_fn = lambda img, i: scan_effect(
                    img, strength=opts["scan_strength"], seed=base_seed + i)

            def on_block(i, n, blk):
                self.after(0, self._set_progress, i, n, blk.get("type", "?"))

            def on_error(i, blk, exc):
                self.after(0, self._log_status,
                          f"Aviso: bloco {i} ({blk.get('type')}) falhou: {exc}")

            out_dir = os.path.dirname(out_base) or "."
            os.makedirs(out_dir, exist_ok=True)
            paths = []
            gen = renderer.iter_document(blocks, xh=opts["xh"], page_w=opts["width"],
                                         ruled=opts["ruled"], on_block=on_block,
                                         on_error=on_error)
            for pi, page in enumerate(gen, 1):
                if scan_fn:
                    page = scan_fn(page, pi)
                pp = f"{out_base}_p{pi}.png"
                page.save(pp)
                paths.append(pp)
                self.after(0, self._update_preview, pp, pi)

            pdf_path = None
            if paths:
                pages = [Image.open(p).convert("RGB") for p in paths]
                pdf_path = f"{out_base}.pdf"
                pages[0].save(pdf_path, save_all=True, append_images=pages[1:])

            self.after(0, self._finish, len(paths), pdf_path, time.time() - t0)
        except Exception as exc:
            self.after(0, self._fail, str(exc))

    # ---- UI callbacks (always run on the main thread via `after`) -------
    def _set_progress(self, i, n, kind):
        self.progress.config(maximum=max(1, n), value=i + 1)
        self.status_var.set(f"[{i + 1}/{n}] {kind}")

    def _log_status(self, msg):
        self.status_var.set(msg)

    def _update_preview(self, path, page_no):
        img = Image.open(path)
        img.thumbnail((360, 480))
        self._preview_imgtk = ImageTk.PhotoImage(img)
        self.preview_label.config(image=self._preview_imgtk)
        self.status_var.set(f"Pagina {page_no} pronta...")

    def _finish(self, n_pages, pdf_path, elapsed):
        self._busy = False
        self.generate_btn.config(state="normal")
        self.status_var.set(f"Concluido: {n_pages} pagina(s) em {elapsed:.1f}s")
        if pdf_path:
            messagebox.showinfo("Concluido",
                               f"{n_pages} pagina(s) geradas em {elapsed:.1f}s.\nPDF: {pdf_path}")
        else:
            messagebox.showwarning("Nada gerado", "Nenhuma pagina foi produzida.")

    def _fail(self, msg):
        self._busy = False
        self.generate_btn.config(state="normal")
        self.status_var.set("Erro.")
        messagebox.showerror("Erro", msg)

    def _on_close(self):
        if self._busy and not messagebox.askyesno(
                "Gerando...", "Uma geracao esta em andamento. Fechar mesmo assim?"):
            return
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
