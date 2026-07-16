"""Desktop front-end for the classical image filters, built with
customtkinter for a modern look (rounded cards, sliders, a before/after
preview) instead of stock Tkinter's raised-panel side-menu look.

Load an image, tune brightness/contrast, pick a convolution kernel preset,
add Gaussian noise, and/or run the (now vectorized) Kuwahara filter, preview
the result next to the original, and save it -- all from one window instead
of switching between five separate panels with no way to export anything.

Keeps the good bones of the original app (interactiveinterface.py): a side
list of options and per-effect controls -- but wired to the vectorized
imgfilters/ package, with a save dialog, a live before/after preview, and
Kuwahara/large-image runs happening on a background thread with a progress
indicator so the window never freezes.

    python filters_gui.py
"""
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image

from imgfilters.io import load_image, save_image
from imgfilters.pointops import adjust_brightness_contrast, add_gaussian_noise
from imgfilters.convolution import convolution_filter, KERNELS
from imgfilters.kuwahara import kuwahara_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ICON_PATH = os.path.join(HERE, "assets", "icon.ico")
LOGO_PATH = os.path.join(HERE, "assets", "logo.png")

# brand palette -- same one used by the sibling handwritten_text project
INK = "#141E3C"
INK_HOVER = "#232B4A"
PAPER = "#FCFAF4"
CARD = "#F3F0E7"
BORDER = "#E1DCCB"
MUTED = "#8A8577"

# label -> internal kernel-preset key, in the same order as the original
# combobox (Blur 3x3, Horizontal/Vertical Derivative, Sobel Horizontal/Vertical)
KERNEL_CHOICES = [
    ("Blur 3x3", "blur3x3"),
    ("Derivada H", "horizontal-derivative"),
    ("Derivada V", "vertical-derivative"),
    ("Sobel H", "sobel-h"),
    ("Sobel V", "sobel-v"),
]

PREVIEW_MAX = 380


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("light")
        self.title("Classical Filters")
        self.geometry("1180x800")
        self.minsize(940, 640)
        self.configure(fg_color=PAPER)
        if os.path.exists(ICON_PATH):
            try:
                self.iconbitmap(ICON_PATH)
            except Exception:
                pass

        self._logo_ctkimage = None
        self._before_ctkimage = None
        self._after_ctkimage = None
        self._busy = False
        self._debounce_id = None          # pending self.after(...) recompute
        self._recompute_generation = 0    # guards against stale async Kuwahara results

        self.original_image = None   # BGR uint8 numpy array, as loaded
        self.result_image = None     # BGR uint8 numpy array, live pipeline result
        self.image_path = None

        self._fonts()
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- fonts --------------------------------------------------------
    def _fonts(self):
        self.f_title = ctk.CTkFont(family="Segoe UI", size=26, weight="bold")
        self.f_subtitle = ctk.CTkFont(family="Segoe UI", size=12)
        self.f_section = ctk.CTkFont(family="Segoe UI", size=13, weight="bold")
        self.f_body = ctk.CTkFont(family="Segoe UI", size=13)
        self.f_small = ctk.CTkFont(family="Segoe UI", size=11)
        self.f_button = ctk.CTkFont(family="Segoe UI", size=14, weight="bold")

    # ---- layout ---------------------------------------------------------
    def _build(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_header()

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=16, pady=(8, 8))
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=0)
        body.grid_rowconfigure(0, weight=1)

        self._build_preview(body)
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
        ctk.CTkLabel(titles, text="Classical Filters", font=self.f_title,
                    text_color=INK).pack(anchor="w")
        ctk.CTkLabel(titles, text="filtros classicos de imagem, vetorizados",
                    font=self.f_subtitle, text_color=MUTED).pack(anchor="w")

        sep = ctk.CTkFrame(self, fg_color=BORDER, height=1, corner_radius=0)
        sep.grid(row=0, column=0, sticky="sew")

    def _build_preview(self, parent):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=14,
                            border_width=1, border_color=BORDER)
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        card.grid_columnconfigure(0, weight=1)
        card.grid_columnconfigure(1, weight=1)
        card.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(card, text="Antes", font=self.f_section, text_color=INK
                    ).grid(row=0, column=0, pady=(12, 4))
        ctk.CTkLabel(card, text="Depois", font=self.f_section, text_color=INK
                    ).grid(row=0, column=1, pady=(12, 4))

        self.before_label = ctk.CTkLabel(
            card, text="Carregue uma imagem para comecar",
            font=self.f_small, text_color=MUTED, fg_color=PAPER,
            corner_radius=10)
        self.before_label.grid(row=1, column=0, sticky="nsew", padx=(14, 7), pady=(0, 14))

        self.after_label = ctk.CTkLabel(
            card, text="Ative um efeito ao lado para ver o resultado aqui",
            font=self.f_small, text_color=MUTED, fg_color=PAPER,
            corner_radius=10)
        self.after_label.grid(row=1, column=1, sticky="nsew", padx=(7, 14), pady=(0, 14))

    def _build_sidebar(self, parent):
        side = ctk.CTkScrollableFrame(parent, fg_color="transparent", width=340,
                                      scrollbar_button_color=BORDER,
                                      scrollbar_button_hover_color=MUTED)
        side.grid(row=0, column=1, sticky="nsew")
        side.grid_columnconfigure(0, weight=1)

        self._build_image_card(side)
        self._build_adjust_card(side)
        self._build_conv_card(side)
        self._build_noise_card(side)
        self._build_kuwahara_card(side)

    def _card(self, parent, title):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=14,
                            border_width=1, border_color=BORDER)
        card.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(card, text=title, font=self.f_section, text_color=INK
                    ).pack(anchor="w", padx=14, pady=(12, 4))
        return card

    def _slider(self, parent, label, var, lo, hi, fmt="{:.2f}", enable_var=None):
        """Pairs a CTkSlider with a live-updating value label -- same helper
        pattern as the sibling handwritten_text GUI."""
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
            if enable_var is not None:
                # touching a slider means the user wants that effect active
                # right now -- no separate "turn the switch on first" step.
                enable_var.set(True)
            self._schedule_recompute()

        slider = ctk.CTkSlider(row, from_=lo, to=hi, command=on_change,
                               fg_color=BORDER, progress_color=INK,
                               button_color=INK, button_hover_color=INK_HOVER)
        slider.set(var.get())
        slider.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        # CTkSlider has no bound `variable=`, so a plain var.set(...) during
        # _reset()/_load_image() would leave the widget's handle/label stuck
        # on the old position -- stash the label + format here so _set_slider
        # can drive the widget itself back to the default too.
        slider.value_label = value_lbl
        slider.value_fmt = fmt
        return slider

    def _set_slider(self, slider, var, value):
        """Resets both the bound Var and the CTkSlider widget + its live
        value label -- setting the Var alone does not move the widget."""
        var.set(value)
        slider.set(value)
        slider.value_label.configure(text=slider.value_fmt.format(value))

    def _build_image_card(self, parent):
        card = self._card(parent, "Imagem")
        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=(4, 12))
        ctk.CTkButton(row, text="Carregar...", command=self._load_image,
                     fg_color=INK, hover_color=INK_HOVER, text_color=PAPER,
                     font=self.f_body).pack(side="left", padx=(0, 6))
        ctk.CTkButton(row, text="Salvar como...", command=self._save_result,
                     fg_color=CARD, hover_color=BORDER, text_color=INK,
                     border_width=1, border_color=BORDER, font=self.f_body
                     ).pack(side="left", padx=6)
        ctk.CTkButton(row, text="Resetar", command=self._reset,
                     fg_color=CARD, hover_color=BORDER, text_color=INK,
                     border_width=1, border_color=BORDER, font=self.f_body
                     ).pack(side="left", padx=6)

    def _enable_switch(self, card, text, var):
        """An ON/OFF switch (default OFF) that gates whether a card's effect
        participates in the live pipeline -- toggling it (like moving any
        slider) triggers a debounced recompute from the original image."""
        ctk.CTkSwitch(card, text=text, variable=var, onvalue=True, offvalue=False,
                     command=self._schedule_recompute,
                     font=self.f_body, text_color=INK, progress_color=INK,
                     button_color=PAPER, button_hover_color=PAPER
                     ).pack(anchor="w", padx=14, pady=(0, 6))

    def _build_adjust_card(self, parent):
        card = self._card(parent, "Ajuste (brilho/contraste)")
        self.adjust_enabled_var = tk.BooleanVar(value=False)
        self._enable_switch(card, "Ativar ajuste de brilho/contraste", self.adjust_enabled_var)
        self.beta_var = tk.DoubleVar(value=0.0)
        self._beta_slider = self._slider(card, "Brilho (beta)", self.beta_var, -100, 100, fmt="{:.0f}",
                                         enable_var=self.adjust_enabled_var)
        self.k_var = tk.DoubleVar(value=1.0)
        self._k_slider = self._slider(card, "Contraste (k)", self.k_var, 0.0, 3.0,
                                      enable_var=self.adjust_enabled_var)
        ctk.CTkFrame(card, fg_color="transparent", height=1).pack(fill="x", pady=(0, 8))

    def _build_conv_card(self, parent):
        card = self._card(parent, "Convolucao")

        self.conv_enabled_var = tk.BooleanVar(value=False)
        self._enable_switch(card, "Ativar convolucao", self.conv_enabled_var)

        # CTkSegmentedButton shares one text_color across selected/unselected
        # states, which made the selected segment's text invisible against
        # its own selected_color background -- plain CTkButtons with
        # per-state fg_color/text_color (same fix as handwrite_gui.py's
        # mode selector) avoid that bug entirely.
        self.kernel_var = tk.StringVar(value=KERNEL_CHOICES[0][1])
        row1 = ctk.CTkFrame(card, fg_color="transparent")
        row1.pack(fill="x", padx=14, pady=(2, 2))
        row2 = ctk.CTkFrame(card, fg_color="transparent")
        row2.pack(fill="x", padx=14, pady=(2, 6))
        self._kernel_buttons = {}
        for i, (label, key) in enumerate(KERNEL_CHOICES):
            target_row = row1 if i < 3 else row2
            btn = ctk.CTkButton(target_row, text=label, corner_radius=8,
                                font=self.f_small, width=90, border_width=1,
                                border_color=BORDER,
                                command=lambda k=key: self._select_kernel(k))
            btn.pack(side="left", padx=3)
            self._kernel_buttons[key] = btn
        self._refresh_kernel_buttons()

        self.keep_color_var = tk.BooleanVar(value=False)
        ctk.CTkSwitch(card, text="Manter cor (aplicar por canal)",
                     variable=self.keep_color_var, onvalue=True, offvalue=False,
                     command=self._on_keep_color_change,
                     font=self.f_body, text_color=INK, progress_color=INK,
                     button_color=PAPER, button_hover_color=PAPER
                     ).pack(anchor="w", padx=14, pady=(4, 12))

    def _build_noise_card(self, parent):
        card = self._card(parent, "Ruido gaussiano")
        self.noise_enabled_var = tk.BooleanVar(value=False)
        self._enable_switch(card, "Ativar ruido", self.noise_enabled_var)
        self.noise_var = tk.DoubleVar(value=20.0)
        self._noise_slider = self._slider(card, "Desvio padrao", self.noise_var, 0, 100, fmt="{:.0f}",
                                          enable_var=self.noise_enabled_var)
        ctk.CTkFrame(card, fg_color="transparent", height=1).pack(fill="x", pady=(0, 8))

    def _build_kuwahara_card(self, parent):
        card = self._card(parent, "Kuwahara")
        self.kuwahara_enabled_var = tk.BooleanVar(value=False)
        self._enable_switch(card, "Ativar kuwahara", self.kuwahara_enabled_var)
        ctk.CTkLabel(card,
                    text="Vetorizado -- sem limite artificial de janela",
                    font=self.f_small, text_color=MUTED
                    ).pack(anchor="w", padx=14, pady=(0, 4))
        self.kuwahara_var = tk.DoubleVar(value=5.0)
        self._kuwahara_slider = self._slider(card, "Tamanho da janela", self.kuwahara_var, 3, 25, fmt="{:.0f}",
                                             enable_var=self.kuwahara_enabled_var)
        ctk.CTkFrame(card, fg_color="transparent", height=1).pack(fill="x", pady=(0, 8))

    def _build_footer(self):
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 14))

        self.progress = ctk.CTkProgressBar(bar, width=320, progress_color=INK,
                                           fg_color=CARD)
        self.progress.set(0)
        self.progress.pack(side="left")

        self.status_var = tk.StringVar(value="Pronto.")
        ctk.CTkLabel(bar, textvariable=self.status_var, font=self.f_body,
                    text_color=MUTED).pack(side="left", padx=14)

    # ---- kernel button styling -------------------------------------------
    def _select_kernel(self, key):
        self.kernel_var.set(key)
        self.conv_enabled_var.set(True)
        self._refresh_kernel_buttons()
        self._schedule_recompute()

    def _on_keep_color_change(self):
        self.conv_enabled_var.set(True)
        self._schedule_recompute()

    def _refresh_kernel_buttons(self):
        current = self.kernel_var.get()
        for key, btn in self._kernel_buttons.items():
            if key == current:
                btn.configure(fg_color=INK, hover_color=INK_HOVER, text_color=PAPER)
            else:
                btn.configure(fg_color=CARD, hover_color=BORDER, text_color=INK)

    # ---- image I/O --------------------------------------------------------
    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Carregar imagem",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp"), ("Todos os arquivos", "*.*")])
        if not path:
            return
        try:
            image = load_image(path)
        except (FileNotFoundError, ValueError) as exc:
            messagebox.showerror("Erro ao carregar", str(exc))
            return
        self.image_path = path
        self.original_image = image
        self.result_image = image
        # loading a new image is a fresh start -- same full switch/slider
        # reset as _reset(), just applied to the newly loaded image.
        self._apply_defaults()
        self._update_preview(self.before_label, image)
        self._update_preview(self.after_label, image)
        self.status_var.set(f"Carregado: {os.path.basename(path)} ({image.shape[1]}x{image.shape[0]})")

    def _apply_defaults(self):
        """Resets every enable-switch and slider to its initial value, and
        invalidates any pending/in-flight recompute so a stale debounced
        call or a stale Kuwahara thread can't clobber the reset state."""
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
            self._debounce_id = None
        self._recompute_generation += 1
        self._busy = False
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress.set(0)

        self.adjust_enabled_var.set(False)
        self.conv_enabled_var.set(False)
        self.noise_enabled_var.set(False)
        self.kuwahara_enabled_var.set(False)

        self._set_slider(self._beta_slider, self.beta_var, 0.0)
        self._set_slider(self._k_slider, self.k_var, 1.0)
        self._set_slider(self._noise_slider, self.noise_var, 20.0)
        self._set_slider(self._kuwahara_slider, self.kuwahara_var, 5.0)

        self.kernel_var.set(KERNEL_CHOICES[0][1])
        self._refresh_kernel_buttons()
        self.keep_color_var.set(False)

    def _reset(self):
        if self.original_image is None:
            return
        self._apply_defaults()
        self.result_image = self.original_image
        self._update_preview(self.after_label, self.result_image)
        self.status_var.set("Resetado para a imagem original.")

    def _save_result(self):
        if self.result_image is None:
            messagebox.showwarning("Aviso", "Nao ha imagem para salvar ainda.")
            return
        path = filedialog.asksaveasfilename(
            title="Salvar como", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")])
        if not path:
            return
        try:
            save_image(self.result_image, path)
        except ValueError as exc:
            messagebox.showerror("Erro ao salvar", str(exc))
            return
        self.status_var.set(f"Salvo: {path}")

    def _update_preview(self, label, image_bgr):
        rgb = image_bgr[:, :, ::-1]
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((PREVIEW_MAX, PREVIEW_MAX))
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
        if label is self.before_label:
            self._before_ctkimage = ctk_img
        else:
            self._after_ctkimage = ctk_img
        label.configure(image=ctk_img, text="")

    # ---- live pipeline ------------------------------------------------------
    def _schedule_recompute(self):
        """Debounces recomputes so a slider drag (or a burst of control
        changes) coalesces into a single recompute ~250ms after things
        settle, instead of spawning a pile of overlapping Kuwahara threads."""
        if self._debounce_id is not None:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(250, self._recompute)

    def _active_effects_summary(self):
        labels = []
        if self.adjust_enabled_var.get():
            labels.append("ajuste")
        if self.conv_enabled_var.get():
            labels.append("convolucao")
        if self.noise_enabled_var.get():
            labels.append("ruido")
        if self.kuwahara_enabled_var.get():
            labels.append("kuwahara")
        return ", ".join(labels)

    def _set_idle_status(self):
        summary = self._active_effects_summary()
        self.status_var.set(f"Ativo: {summary}" if summary
                            else "Nenhum efeito ativo (mostrando original)")

    def _recompute(self):
        """Rebuilds the result from scratch, starting at self.original_image
        and applying only the currently-enabled effects in the fixed
        adjust -> conv -> noise -> kuwahara order (same order as the CLI).
        This is what makes moving a slider always show that effect applied
        to the ORIGINAL (plus whatever else is on) instead of compounding
        onto whatever the last click happened to produce."""
        self._debounce_id = None
        if self.original_image is None:
            return

        # Every recompute invalidates any previous one -- including a
        # Kuwahara thread that may still be running -- so a stale result
        # (e.g. from a run that started before the user flipped Kuwahara
        # off, or before a newer edit superseded it) never clobbers newer
        # state when it lands late.
        self._recompute_generation += 1
        generation = self._recompute_generation

        image = self.original_image
        try:
            if self.adjust_enabled_var.get():
                image = adjust_brightness_contrast(image, self.beta_var.get(), self.k_var.get())
            if self.conv_enabled_var.get():
                kernel = KERNELS[self.kernel_var.get()]
                image = convolution_filter(image, kernel, keep_color=self.keep_color_var.get())
            if self.noise_enabled_var.get():
                image = add_gaussian_noise(image, self.noise_var.get())
        except Exception as exc:
            messagebox.showerror("Erro ao aplicar filtro", str(exc))
            return

        if self.kuwahara_enabled_var.get():
            # still O(pixels) even vectorized -- run off the main thread with
            # a progress indicator so the window doesn't appear to freeze.
            self._busy = True
            self.progress.configure(mode="indeterminate")
            self.progress.start()
            self.status_var.set("Processando Kuwahara...")
            window_size = int(self.kuwahara_var.get())
            threading.Thread(target=self._run_kuwahara, args=(image, window_size, generation),
                             daemon=True).start()
            return

        self._busy = False
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress.set(0)
        self.result_image = image
        self._update_preview(self.after_label, image)
        self._set_idle_status()

    def _run_kuwahara(self, source, window_size, generation):
        """Runs in a background thread -- never touch widgets directly here,
        only via self.after(...) so Tk stays on the main thread. `source` is
        already the adjust/conv/noise-processed image, not the raw original."""
        try:
            result = kuwahara_filter(source, window_size)
        except Exception as exc:
            self.after(0, self._kuwahara_failed, str(exc), generation)
            return
        self.after(0, self._kuwahara_done, result, generation)

    def _kuwahara_done(self, result, generation):
        if generation != self._recompute_generation:
            return  # superseded by a newer recompute -- discard silently
        self._busy = False
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress.set(0)
        self.result_image = result
        self._update_preview(self.after_label, result)
        self._set_idle_status()

    def _kuwahara_failed(self, msg, generation):
        if generation != self._recompute_generation:
            return  # superseded -- a newer recompute already took over
        self._busy = False
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress.set(0)
        self.status_var.set("Erro.")
        messagebox.showerror("Erro ao aplicar Kuwahara", msg)

    def _on_close(self):
        if self._busy and not messagebox.askyesno(
                "Processando...", "Um filtro ainda esta sendo aplicado. Fechar mesmo assim?"):
            return
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
