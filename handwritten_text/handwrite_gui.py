"""
Desktop front-end for the handwriting pipeline, built with customtkinter for
a modern look (rounded cards, switches/sliders, a live page preview) instead
of stock Tkinter.

Type Markdown+LaTeX / plain LaTeX / plain text directly, or import a
.md/.tex/.txt file, tune the same options the CLIs expose, and generate a
lazily-streamed PNG-per-page + PDF using the current hw/ engine
(iter_document, --hand-math, ink texture, paper scan, etc.).

Keeps the good bones of this project's original Tkinter app (app.py): a
background thread so the window never freezes, a progress callback, a
save-file dialog -- but wired to the new engine, and considerably more
polished visually (branded palette, card layout, icon, live preview).

  python handwrite_gui.py
"""
import os
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image

from hw.render import HandwritingRenderer
from hw import markdown_render, latex_render

HERE = os.path.dirname(os.path.abspath(__file__))
ICON_PATH = os.path.join(HERE, "assets", "icon.ico")
LOGO_PATH = os.path.join(HERE, "assets", "logo.png")

# brand palette -- lifted straight from the renderer's own ink/paper colors
INK = "#141E3C"
INK_SOFT = "#333B5C"
INK_HOVER = "#232B4A"
PAPER = "#FCFAF4"
CARD = "#F3F0E7"
BORDER = "#E1DCCB"
MUTED = "#8A8577"

MODES = ("Markdown + LaTeX", "LaTeX puro", "Texto simples")

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


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("light")
        self.title("Neural Handwriting")
        self.geometry("1200x820")
        self.minsize(920, 640)
        self.configure(fg_color=PAPER)
        if os.path.exists(ICON_PATH):
            try:
                self.iconbitmap(ICON_PATH)
            except Exception:
                pass

        self._logo_ctkimage = None
        self._preview_ctkimage = None
        self._busy = False

        self._fonts()
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- fonts ------------------------------------------------------------
    def _fonts(self):
        self.f_title = ctk.CTkFont(family="Segoe Script", size=30)
        self.f_subtitle = ctk.CTkFont(family="Segoe UI", size=12)
        self.f_section = ctk.CTkFont(family="Segoe UI", size=13, weight="bold")
        self.f_body = ctk.CTkFont(family="Segoe UI", size=13)
        self.f_small = ctk.CTkFont(family="Segoe UI", size=11)
        self.f_mono = ctk.CTkFont(family="Consolas", size=13)
        self.f_button = ctk.CTkFont(family="Segoe UI", size=15, weight="bold")

    # ---- layout -------------------------------------------------------
    def _build(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=16, pady=(8, 8))
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=0)
        body.grid_rowconfigure(1, weight=1)

        self._build_toolbar(body)
        self._build_editor(body)
        self._build_sidebar(body)
        self._build_footer()

    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=PAPER, corner_radius=0,
                              border_width=0, height=72)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)

        if os.path.exists(LOGO_PATH):
            logo_img = Image.open(LOGO_PATH)
            self._logo_ctkimage = ctk.CTkImage(light_image=logo_img,
                                               dark_image=logo_img, size=(44, 44))
            ctk.CTkLabel(header, text="", image=self._logo_ctkimage
                        ).pack(side="left", padx=(18, 8), pady=10)

        titles = ctk.CTkFrame(header, fg_color="transparent")
        titles.pack(side="left", pady=6)
        ctk.CTkLabel(titles, text="Neural Handwriting", font=self.f_title,
                    text_color=INK).pack(anchor="w")
        ctk.CTkLabel(titles, text="escreva qualquer coisa (e LaTeX) com a sua letra",
                    font=self.f_subtitle, text_color=MUTED).pack(anchor="w")

        sep = ctk.CTkFrame(self, fg_color=BORDER, height=1, corner_radius=0)
        sep.grid(row=0, column=0, sticky="sew")

    def _build_toolbar(self, parent):
        bar = ctk.CTkFrame(parent, fg_color="transparent")
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ctk.CTkLabel(bar, text="Conteudo", font=self.f_section,
                    text_color=INK).pack(side="left", padx=(2, 8))
        self.mode_var = tk.StringVar(value=MODES[0])
        # CTkSegmentedButton only exposes a single text_color for every
        # segment, selected or not -- with a dark selected_color that made
        # the selected label's text the same dark navy as its own background
        # (invisible). Plain CTkButtons give full control over both states.
        mode_row = ctk.CTkFrame(bar, fg_color="transparent")
        mode_row.pack(side="left", padx=6)
        self._mode_buttons = {}
        for m in MODES:
            btn = ctk.CTkButton(mode_row, text=m, corner_radius=8, font=self.f_body,
                                border_width=1, border_color=BORDER,
                                command=lambda mm=m: self._select_mode(mm))
            btn.pack(side="left", padx=3)
            self._mode_buttons[m] = btn
        self._refresh_mode_buttons()

        ctk.CTkButton(bar, text="Importar arquivo...", command=self._import_file,
                     fg_color=CARD, hover_color=BORDER, text_color=INK,
                     border_width=1, border_color=BORDER, font=self.f_body
                     ).pack(side="left", padx=6)
        ctk.CTkButton(bar, text="Limpar", command=self._clear_text,
                     fg_color=CARD, hover_color=BORDER, text_color=INK,
                     border_width=1, border_color=BORDER, font=self.f_body
                     ).pack(side="left", padx=6)

    def _build_editor(self, parent):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=14,
                            border_width=1, border_color=BORDER)
        card.grid(row=1, column=0, sticky="nsew", padx=(0, 12))
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(0, weight=1)

        self.text = ctk.CTkTextbox(card, wrap="word", fg_color=CARD,
                                   text_color=INK, font=self.f_mono,
                                   border_width=0, corner_radius=12)
        self.text.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.text.insert("1.0", DEMO_TEXT)

    def _build_sidebar(self, parent):
        side = ctk.CTkScrollableFrame(parent, fg_color="transparent", width=340,
                                      scrollbar_button_color=BORDER,
                                      scrollbar_button_hover_color=MUTED)
        side.grid(row=1, column=1, sticky="nsew")
        side.grid_columnconfigure(0, weight=1)

        self._build_options_card(side)
        self._build_advanced_card(side)
        self._build_preview_card(side)

    def _card(self, parent, title):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=14,
                            border_width=1, border_color=BORDER)
        card.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(card, text=title, font=self.f_section, text_color=INK
                    ).pack(anchor="w", padx=14, pady=(12, 4))
        return card

    def _switch(self, parent, text, var, default=True):
        var.set(default)
        ctk.CTkSwitch(parent, text=text, variable=var, onvalue=True, offvalue=False,
                     font=self.f_body, text_color=INK, progress_color=INK,
                     button_color=PAPER, button_hover_color=PAPER
                     ).pack(anchor="w", padx=14, pady=4)

    def _slider(self, parent, label, var, lo, hi, fmt="{:.2f}"):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=4)
        row.grid_columnconfigure(0, weight=1)
        top = ctk.CTkFrame(row, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(top, text=label, font=self.f_small, text_color=MUTED
                    ).pack(side="left")
        value_lbl = ctk.CTkLabel(top, text=fmt.format(var.get()),
                                 font=self.f_small, text_color=INK)
        value_lbl.pack(side="right")

        def on_change(v):
            var.set(float(v))
            value_lbl.configure(text=fmt.format(float(v)))

        slider = ctk.CTkSlider(row, from_=lo, to=hi, command=on_change,
                               fg_color=BORDER, progress_color=INK,
                               button_color=INK, button_hover_color=INK_HOVER)
        slider.set(var.get())
        slider.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        return slider

    def _build_options_card(self, parent):
        card = self._card(parent, "Opcoes")
        self.hand_math_var = tk.BooleanVar()
        self._switch(card, "Matematica na minha letra", self.hand_math_var, True)
        self.scan_var = tk.BooleanVar()
        self._switch(card, "Papel escaneado", self.scan_var, True)
        self.ruled_var = tk.BooleanVar()
        self._switch(card, "Linhas de caderno", self.ruled_var, False)
        self.model_var = tk.BooleanVar()
        self._switch(card, "Rede neural p/ letras faltantes", self.model_var, False)
        self.page_numbers_var = tk.BooleanVar(value=True)
        self._switch(card, "Numerar paginas", self.page_numbers_var, True)

        self.scan_strength = tk.DoubleVar(value=1.0)
        self._slider(card, "Intensidade do papel", self.scan_strength, 0.2, 2.0)

        title_row = ctk.CTkFrame(card, fg_color="transparent")
        title_row.pack(fill="x", padx=14, pady=(6, 4))
        ctk.CTkLabel(title_row, text="Titulo do documento (opcional)",
                    font=self.f_small, text_color=MUTED).pack(anchor="w")
        self.title_var = tk.StringVar(value="")
        ctk.CTkEntry(title_row, textvariable=self.title_var,
                    fg_color=PAPER, text_color=INK, border_color=BORDER
                    ).pack(fill="x", pady=(2, 0))

        seed_row = ctk.CTkFrame(card, fg_color="transparent")
        seed_row.pack(fill="x", padx=14, pady=(6, 12))
        ctk.CTkLabel(seed_row, text="Semente (vazio = aleatoria)",
                    font=self.f_small, text_color=MUTED).pack(side="left")
        self.seed_var = tk.StringVar(value="")
        ctk.CTkEntry(seed_row, textvariable=self.seed_var, width=70,
                    fg_color=PAPER, text_color=INK, border_color=BORDER
                    ).pack(side="right")

    def _build_advanced_card(self, parent):
        card = self._card(parent, "Ajustes finos")
        self.ink_var = tk.DoubleVar(value=0.8)
        self._slider(card, "Tinta", self.ink_var, 0.0, 1.5)
        self.tremor_var = tk.DoubleVar(value=0.3)
        self._slider(card, "Tremor da letra", self.tremor_var, 0.0, 1.5)
        self.regularize_var = tk.DoubleVar(value=1.0)
        self._slider(card, "Regularizar espessura", self.regularize_var, 0.0, 1.0)
        self.stroke_var = tk.DoubleVar(value=0.11)
        self._slider(card, "Espessura do traco", self.stroke_var, 0.05, 0.20)
        self.math_style_var = tk.DoubleVar(value=1.0)
        self._slider(card, "Estilo da matematica", self.math_style_var, 0.0, 2.0)
        self.xh_var = tk.DoubleVar(value=26)
        self._slider(card, "Tamanho da letra (px)", self.xh_var, 14, 60, fmt="{:.0f}")
        self.width_var = tk.DoubleVar(value=1000)
        self._slider(card, "Largura da pagina (px)", self.width_var, 500, 2000, fmt="{:.0f}")
        ctk.CTkLabel(card, text="", height=4, fg_color="transparent").pack()

    def _build_preview_card(self, parent):
        card = self._card(parent, "Pre-visualizacao")
        self.preview_label = ctk.CTkLabel(
            card, text="A primeira pagina gerada\naparecera aqui",
            font=self.f_small, text_color=MUTED, fg_color=PAPER,
            corner_radius=10, width=280, height=360)
        self.preview_label.pack(padx=14, pady=(4, 14))

    def _build_footer(self):
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 14))

        self.generate_btn = ctk.CTkButton(
            bar, text="Gerar", command=self._start_generation, width=140, height=40,
            fg_color=INK, hover_color=INK_HOVER, text_color=PAPER,
            font=self.f_button, corner_radius=10)
        self.generate_btn.pack(side="left")

        self.progress = ctk.CTkProgressBar(bar, width=320, progress_color=INK,
                                           fg_color=CARD)
        self.progress.set(0)
        self.progress.pack(side="left", padx=14)

        self.status_var = tk.StringVar(value="Pronto.")
        ctk.CTkLabel(bar, textvariable=self.status_var, font=self.f_body,
                    text_color=MUTED).pack(side="left", padx=6)

    # ---- actions --------------------------------------------------------
    def _select_mode(self, mode):
        self.mode_var.set(mode)
        self._refresh_mode_buttons()

    def _refresh_mode_buttons(self):
        current = self.mode_var.get()
        for m, btn in self._mode_buttons.items():
            if m == current:
                btn.configure(fg_color=INK, hover_color=INK_HOVER, text_color=PAPER)
            else:
                btn.configure(fg_color=CARD, hover_color=BORDER, text_color=INK)

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
        self._select_mode(MODES[0] if ext == ".md" else
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
            xh=int(self.xh_var.get()), width=int(self.width_var.get()), seed=seed,
            title=self.title_var.get().strip() or None,
            page_numbers=self.page_numbers_var.get(),
        )

        self._busy = True
        self.generate_btn.configure(state="disabled")
        self.progress.set(0)
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
            MARGIN = 70
            paths = []
            gen = renderer.iter_document(blocks, xh=opts["xh"], page_w=opts["width"],
                                         margin=MARGIN, ruled=opts["ruled"],
                                         on_block=on_block, on_error=on_error)
            for pi, page in enumerate(gen, 1):
                if opts["title"] or opts["page_numbers"]:
                    # stamped BEFORE the paper-scan warp, so the title/page
                    # number distorts along with the rest of the page instead
                    # of looking like a crisp overlay pasted onto a warped scan
                    renderer.stamp_header_footer(
                        page, margin=MARGIN, xh=opts["xh"], title=opts["title"],
                        page_num=pi if opts["page_numbers"] else None)
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
        self.progress.set((i + 1) / max(1, n))
        self.status_var.set(f"[{i + 1}/{n}] {kind}")

    def _log_status(self, msg):
        self.status_var.set(msg)

    def _update_preview(self, path, page_no):
        img = Image.open(path)
        img.thumbnail((280, 396))
        self._preview_ctkimage = ctk.CTkImage(light_image=img, dark_image=img,
                                              size=img.size)
        self.preview_label.configure(image=self._preview_ctkimage, text="")
        self.status_var.set(f"Pagina {page_no} pronta...")

    def _finish(self, n_pages, pdf_path, elapsed):
        self._busy = False
        self.generate_btn.configure(state="normal")
        self.status_var.set(f"Concluido: {n_pages} pagina(s) em {elapsed:.1f}s")
        if pdf_path:
            messagebox.showinfo("Concluido",
                               f"{n_pages} pagina(s) geradas em {elapsed:.1f}s.\nPDF: {pdf_path}")
        else:
            messagebox.showwarning("Nada gerado", "Nenhuma pagina foi produzida.")

    def _fail(self, msg):
        self._busy = False
        self.generate_btn.configure(state="normal")
        self.status_var.set("Erro.")
        messagebox.showerror("Erro", msg)

    def _on_close(self):
        if self._busy and not messagebox.askyesno(
                "Gerando...", "Uma geracao esta em andamento. Fechar mesmo assim?"):
            return
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
