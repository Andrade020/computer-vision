"""Desktop front-end for the audio DSP pipeline, built with customtkinter for
a modern look (rounded cards, sliders, an embedded spectrum plot) instead of
stock Tkinter.

Load a .wav/.flac/etc file, toggle the same effects the CLI (audioprocess.py)
exposes on/off with live sliders, and A/B compare the original vs the
processed audio through real transport controls (play/pause, stop, seek) --
something the original prototype's UI never offered even though it kept a
copy of the original audio around unused.

Effects are non-destructive and auto-apply: each has an ON/OFF switch, but
you don't need to flip it before touching its sliders -- moving a slider
turns that effect on by itself. Either way, any change schedules a debounced
recompute that rebuilds the processed buffer from the untouched original by
applying only the enabled effects, in the fixed order trim -> compress ->
echo -> reverb (matching the CLI), on a background thread (reverb on long
clips can be slow) so the window never freezes. The spectrum plot redraws
automatically after every recompute, and if audio was playing when you
tweak a control (or flip Original/Processado), playback keeps going at the
same position on the new buffer instead of stopping.

Keeps the good bones of the original app.py (a single window with
sidebar-selected panels) but wired to the new audiodsp/ package, with
playback done cross-platform via sounddevice instead of os.startfile.

  python audio_gui.py
"""
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import customtkinter as ctk
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from audiodsp import io as audio_io
from audiodsp import effects
from audiodsp import spectrum as spectrum_mod
from audiodsp import stft as stft_mod
from audiodsp import filters as filters_mod
from audiodsp import playback

WINDOW_CHOICES = ["hann", "hamming", "blackman", "bartlett"]

HERE = os.path.dirname(os.path.abspath(__file__))
ICON_PATH = os.path.join(HERE, "assets", "icon.ico")
LOGO_PATH = os.path.join(HERE, "assets", "logo.png")

# brand palette -- same one used by the sibling neural-handwriting project,
# reused here for visual consistency across the repo's mini-apps.
INK = "#141E3C"
INK_HOVER = "#232B4A"
PAPER = "#FCFAF4"
CARD = "#F3F0E7"
BORDER = "#E1DCCB"
MUTED = "#8A8577"


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("light")
        self.title("Audio DSP Studio")
        self.geometry("1180x760")
        self.minsize(940, 620)
        self.configure(fg_color=PAPER)
        if os.path.exists(ICON_PATH):
            try:
                self.iconbitmap(ICON_PATH)
            except Exception:
                pass

        self._logo_ctkimage = None
        self._busy = False

        # audio state
        self.audio_original = None   # untouched, as loaded
        self.audio = None            # current processed buffer
        self.sr = None
        self.audio_path = None

        # debounced auto-apply recompute state
        self._recompute_job = None       # self.after() id, for cancellation
        self._recompute_gen = 0          # generation counter -- discards stale
                                          # background results from superseded
                                          # recomputes (same guard pattern as
                                          # any debounced-background-thread UI)

        # transport / playback state
        self.player = playback.Player()
        self._scrub_dragging = False
        self._tick_job = None

        # spectral-editing session state (see _start_spectral_edit) -- None/
        # False whenever no paint session is in progress
        self._edit_active = False
        self._edit_freqs = None
        self._edit_times = None
        self._edit_S = None
        self._edit_mask = None
        self._edit_hop = None
        self._edit_mesh = None
        self._painting = False

        self._fonts()
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- fonts --------------------------------------------------------
    def _fonts(self):
        self.f_title = ctk.CTkFont(family="Segoe UI", size=27, weight="bold")
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
        body.grid_columnconfigure(0, weight=0)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self._build_sidebar(body)
        self._build_main(body)
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
        ctk.CTkLabel(titles, text="Audio DSP Studio", font=self.f_title,
                    text_color=INK).pack(anchor="w")
        ctk.CTkLabel(titles, text="trim, compressao de Fourier, eco, reverb e espectro",
                    font=self.f_subtitle, text_color=MUTED).pack(anchor="w")

        sep = ctk.CTkFrame(self, fg_color=BORDER, height=1, corner_radius=0)
        sep.grid(row=0, column=0, sticky="sew")

    def _build_sidebar(self, parent):
        side = ctk.CTkScrollableFrame(parent, fg_color="transparent", width=360,
                                      scrollbar_button_color=BORDER,
                                      scrollbar_button_hover_color=MUTED)
        side.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        side.grid_columnconfigure(0, weight=1)

        self._build_file_card(side)
        self._build_effects_card(side)
        self._build_eq_card(side)
        self._build_spectrogram_card(side)
        self._build_spectral_edit_card(side)
        self._build_playback_card(side)

    def _card(self, parent, title):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=14,
                            border_width=1, border_color=BORDER)
        card.pack(fill="x", pady=(0, 12))
        ctk.CTkLabel(card, text=title, font=self.f_section, text_color=INK
                    ).pack(anchor="w", padx=14, pady=(12, 4))
        return card

    def _slider(self, parent, label, var, lo, hi, fmt="{:.2f}", on_change_extra=None,
               attr_prefix=None, enable_var=None):
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
            if on_change_extra is not None:
                on_change_extra()

        slider = ctk.CTkSlider(row, from_=lo, to=hi, command=on_change,
                               fg_color=BORDER, progress_color=INK,
                               button_color=INK, button_hover_color=INK_HOVER)
        slider.set(var.get())
        slider.grid(row=1, column=0, sticky="ew", pady=(2, 0))

        if attr_prefix is not None:
            # stash refs so _reset_to_original can restore the visual state
            # (slider handle position + value label text) programmatically
            setattr(self, f"{attr_prefix}_slider", slider)
            setattr(self, f"{attr_prefix}_value_lbl", value_lbl)
        return slider

    def _build_file_card(self, parent):
        card = self._card(parent, "Arquivo")
        self.file_label_var = tk.StringVar(value="Nenhum arquivo carregado")
        ctk.CTkLabel(card, textvariable=self.file_label_var, font=self.f_body,
                    text_color=INK, wraplength=300, justify="left"
                    ).pack(anchor="w", padx=14, pady=(0, 4))
        self.file_info_var = tk.StringVar(value="")
        ctk.CTkLabel(card, textvariable=self.file_info_var, font=self.f_small,
                    text_color=MUTED).pack(anchor="w", padx=14, pady=(0, 8))
        ctk.CTkButton(card, text="Carregar audio...", command=self._load_audio,
                     fg_color=INK, hover_color=INK_HOVER, text_color=PAPER,
                     font=self.f_body).pack(anchor="w", padx=14, pady=(0, 14))

    def _switch_row(self, parent, label, var):
        """An ON/OFF CTkSwitch bound to a tk.BooleanVar, wired to trigger a
        debounced recompute whenever toggled. Used as the per-effect header
        row so each effect can be turned on/off without a dedicated Apply
        button."""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=(4, 0))
        ctk.CTkLabel(row, text=label, font=self.f_small, text_color=MUTED
                    ).pack(side="left")
        switch = ctk.CTkSwitch(row, text="", variable=var, width=36,
                               progress_color=INK, button_color=PAPER,
                               button_hover_color=PAPER,
                               command=self._schedule_recompute)
        switch.pack(side="right")
        return switch

    def _build_effects_card(self, parent):
        card = self._card(parent, "Efeitos")

        self.trim_on_var = tk.BooleanVar(value=False)
        self._switch_row(card, "Trim (segundos)", self.trim_on_var)
        self.trim_var = tk.DoubleVar(value=10.0)
        self._slider(card, "duracao mantida", self.trim_var, 1, 30, fmt="{:.0f}s",
                    on_change_extra=self._schedule_recompute, attr_prefix="trim",
                    enable_var=self.trim_on_var)

        self.compress_on_var = tk.BooleanVar(value=False)
        self._switch_row(card, "Compressao de Fourier", self.compress_on_var)
        self.compress_var = tk.DoubleVar(value=0.5)
        self._slider(card, "fracao mantida", self.compress_var, 0.05, 1.0,
                    on_change_extra=self._schedule_recompute, attr_prefix="compress",
                    enable_var=self.compress_on_var)

        self.echo_on_var = tk.BooleanVar(value=False)
        self._switch_row(card, "Eco", self.echo_on_var)
        self.echo_delay_var = tk.DoubleVar(value=0.5)
        self._slider(card, "atraso (s)", self.echo_delay_var, 0.05, 2.0,
                    on_change_extra=self._schedule_recompute, attr_prefix="echo_delay",
                    enable_var=self.echo_on_var)
        self.echo_gain_var = tk.DoubleVar(value=0.6)
        self._slider(card, "ganho", self.echo_gain_var, 0.0, 1.0,
                    on_change_extra=self._schedule_recompute, attr_prefix="echo_gain",
                    enable_var=self.echo_on_var)

        self.reverb_on_var = tk.BooleanVar(value=False)
        self._switch_row(card, "Reverb", self.reverb_on_var)
        self.reverb_delays_var = tk.DoubleVar(value=10)
        self._slider(card, "numero de ecos", self.reverb_delays_var, 1, 30, fmt="{:.0f}",
                    on_change_extra=self._schedule_recompute, attr_prefix="reverb_delays",
                    enable_var=self.reverb_on_var)
        self.reverb_time_var = tk.DoubleVar(value=0.05)
        self._slider(card, "intervalo (s)", self.reverb_time_var, 0.01, 0.5,
                    on_change_extra=self._schedule_recompute, attr_prefix="reverb_time",
                    enable_var=self.reverb_on_var)

        ctk.CTkButton(card, text="Resetar", command=self._reset_to_original,
                     fg_color=CARD, hover_color=BORDER, text_color=INK,
                     border_width=1, border_color=BORDER, font=self.f_body
                     ).pack(anchor="w", padx=14, pady=(6, 14))

    def _build_eq_card(self, parent):
        """A simple 3-band EQ (graves/medios/agudos) built on real biquad
        filters (audiodsp/filters.py) -- causal/streaming like a real EQ,
        unlike the STFT-based spectral editor below (which can target one
        moment in time but needs the whole recording analyzed up front).
        Runs as the last stage of the effects pipeline, after reverb."""
        card = self._card(parent, "Equalizador")
        ctk.CTkLabel(card,
                    text="EQ real (biquad/IIR), como um equalizador de\n"
                         "verdade -- roda por amostra, nao precisa da\n"
                         "gravacao inteira como o espectrograma.",
                    font=self.f_small, text_color=MUTED, justify="left"
                    ).pack(anchor="w", padx=14, pady=(0, 6))

        self.eq_low_on_var = tk.BooleanVar(value=False)
        self._switch_row(card, "Graves (shelf)", self.eq_low_on_var)
        self.eq_low_freq_var = tk.DoubleVar(value=200.0)
        self._slider(card, "freq (Hz)", self.eq_low_freq_var, 40, 500, fmt="{:.0f}",
                    on_change_extra=self._schedule_recompute, attr_prefix="eq_low_freq",
                    enable_var=self.eq_low_on_var)
        self.eq_low_gain_var = tk.DoubleVar(value=6.0)
        self._slider(card, "ganho (dB)", self.eq_low_gain_var, -18, 18, fmt="{:+.1f}",
                    on_change_extra=self._schedule_recompute, attr_prefix="eq_low_gain",
                    enable_var=self.eq_low_on_var)

        self.eq_mid_on_var = tk.BooleanVar(value=False)
        self._switch_row(card, "Medios (peak)", self.eq_mid_on_var)
        self.eq_mid_freq_var = tk.DoubleVar(value=1000.0)
        self._slider(card, "freq (Hz)", self.eq_mid_freq_var, 200, 5000, fmt="{:.0f}",
                    on_change_extra=self._schedule_recompute, attr_prefix="eq_mid_freq",
                    enable_var=self.eq_mid_on_var)
        self.eq_mid_gain_var = tk.DoubleVar(value=-6.0)
        self._slider(card, "ganho (dB)", self.eq_mid_gain_var, -18, 18, fmt="{:+.1f}",
                    on_change_extra=self._schedule_recompute, attr_prefix="eq_mid_gain",
                    enable_var=self.eq_mid_on_var)

        self.eq_high_on_var = tk.BooleanVar(value=False)
        self._switch_row(card, "Agudos (shelf)", self.eq_high_on_var)
        self.eq_high_freq_var = tk.DoubleVar(value=5000.0)
        self._slider(card, "freq (Hz)", self.eq_high_freq_var, 2000, 15000, fmt="{:.0f}",
                    on_change_extra=self._schedule_recompute, attr_prefix="eq_high_freq",
                    enable_var=self.eq_high_on_var)
        self.eq_high_gain_var = tk.DoubleVar(value=6.0)
        self._slider(card, "ganho (dB)", self.eq_high_gain_var, -18, 18, fmt="{:+.1f}",
                    on_change_extra=self._schedule_recompute, attr_prefix="eq_high_gain",
                    enable_var=self.eq_high_on_var)

    def _current_eq_bands(self):
        bands = []
        if self.eq_low_on_var.get():
            bands.append(dict(type="lowshelf", freq=self.eq_low_freq_var.get(),
                              q=0.707, gain_db=self.eq_low_gain_var.get()))
        if self.eq_mid_on_var.get():
            bands.append(dict(type="peak", freq=self.eq_mid_freq_var.get(),
                              q=1.0, gain_db=self.eq_mid_gain_var.get()))
        if self.eq_high_on_var.get():
            bands.append(dict(type="highshelf", freq=self.eq_high_freq_var.get(),
                              q=0.707, gain_db=self.eq_high_gain_var.get()))
        return bands

    def _build_spectrogram_card(self, parent):
        """Controls for the time-frequency view (see the main panel's
        Espectro/Espectrograma toggle). These only change how the audio is
        *visualized*, never the audio itself -- moving them just redraws
        whichever view is currently on screen, no debounce/recompute needed."""
        card = self._card(parent, "Espectrograma")
        ctk.CTkLabel(card,
                    text="Janela maior = mais preciso em frequencia, mais\n"
                         "borrado no tempo. Janela menor = o oposto.",
                    font=self.f_small, text_color=MUTED, justify="left"
                    ).pack(anchor="w", padx=14, pady=(0, 6))

        self.stft_nfft_var = tk.DoubleVar(value=1024)
        self._slider(card, "tamanho da janela (n_fft)", self.stft_nfft_var, 256, 4096,
                    fmt="{:.0f}", on_change_extra=self._on_spectrogram_settings_change,
                    attr_prefix="stft_nfft")

        self.stft_overlap_var = tk.DoubleVar(value=75.0)
        self._slider(card, "sobreposicao entre janelas", self.stft_overlap_var, 25, 90,
                    fmt="{:.0f}%", on_change_extra=self._on_spectrogram_settings_change,
                    attr_prefix="stft_overlap")

        window_row = ctk.CTkFrame(card, fg_color="transparent")
        window_row.pack(fill="x", padx=14, pady=(2, 12))
        ctk.CTkLabel(window_row, text="janela", font=self.f_small, text_color=MUTED
                    ).pack(side="left")
        self.stft_window_var = tk.StringVar(value="hann")
        self.stft_window_menu = ctk.CTkOptionMenu(
            window_row, values=WINDOW_CHOICES, variable=self.stft_window_var,
            command=lambda _v: self._on_spectrogram_settings_change(),
            fg_color=INK, button_color=INK, button_hover_color=INK_HOVER,
            dropdown_fg_color=CARD, dropdown_text_color=INK,
            text_color=PAPER, font=self.f_small, width=110)
        self.stft_window_menu.pack(side="right")

    def _build_spectral_edit_card(self, parent):
        """A creative, hands-on extension of the spectrogram view: instead of
        only *looking* at the time-frequency plane, paint directly on it to
        erase or boost a patch of it (e.g. "silence this cough between 1.2
        and 1.5 seconds" or "erase everything above 4kHz for this whole
        clip") -- something no purely time-domain effect (trim/echo/reverb)
        can express, since those never see frequency and time at once.

        Editing is a deliberate, one-shot "bake" step (like a paint tool),
        not part of the auto-recompute pipeline the other effects use: you
        Start an edit (snapshots the current audio's STFT), paint, then
        either Aplicar (bakes the result into the current audio) or
        Cancelar (discards the paint session, audio untouched)."""
        card = self._card(parent, "Editor Espectral")
        ctk.CTkLabel(card,
                    text="Pinte no espectrograma para apagar ou realcar um\n"
                         "trecho de tempo/frequencia -- depois aplique para\n"
                         "gerar audio novo a partir da edicao.",
                    font=self.f_small, text_color=MUTED, justify="left"
                    ).pack(anchor="w", padx=14, pady=(0, 6))

        self.paint_mode_var = tk.StringVar(value="Apagar")
        mode_row = ctk.CTkFrame(card, fg_color="transparent")
        mode_row.pack(fill="x", padx=14, pady=(0, 6))
        self._paint_mode_buttons = {}
        for label in ("Apagar", "Realcar"):
            btn = ctk.CTkButton(mode_row, text=label, corner_radius=8,
                                font=self.f_small, border_width=1, border_color=BORDER,
                                command=lambda l=label: self._select_paint_mode(l))
            btn.pack(side="left", expand=True, fill="x", padx=3)
            self._paint_mode_buttons[label] = btn
        self._refresh_paint_mode_buttons()

        self.brush_freq_var = tk.DoubleVar(value=200.0)
        self._slider(card, "pincel (Hz)", self.brush_freq_var, 20, 4000, fmt="{:.0f}")
        self.brush_time_var = tk.DoubleVar(value=120.0)
        self._slider(card, "pincel (ms)", self.brush_time_var, 20, 1000, fmt="{:.0f}")

        actions = ctk.CTkFrame(card, fg_color="transparent")
        actions.pack(fill="x", padx=14, pady=(4, 4))
        ctk.CTkButton(actions, text="Iniciar edicao", command=self._start_spectral_edit,
                     fg_color=INK, hover_color=INK_HOVER, text_color=PAPER,
                     font=self.f_small, width=110).pack(side="left", padx=(0, 4))
        ctk.CTkButton(actions, text="Limpar", command=self._clear_spectral_mask,
                     fg_color=CARD, hover_color=BORDER, text_color=INK,
                     border_width=1, border_color=BORDER, font=self.f_small,
                     width=70).pack(side="left", padx=4)

        actions2 = ctk.CTkFrame(card, fg_color="transparent")
        actions2.pack(fill="x", padx=14, pady=(0, 12))
        ctk.CTkButton(actions2, text="Cancelar", command=self._cancel_spectral_edit,
                     fg_color=CARD, hover_color=BORDER, text_color=INK,
                     border_width=1, border_color=BORDER, font=self.f_small,
                     width=90).pack(side="left", padx=(0, 4))
        ctk.CTkButton(actions2, text="Aplicar", command=self._apply_spectral_edit,
                     fg_color=INK, hover_color=INK_HOVER, text_color=PAPER,
                     font=self.f_small, width=90).pack(side="left", padx=4)

    def _build_playback_card(self, parent):
        card = self._card(parent, "Reproducao")

        # CTkSegmentedButton shares one text_color across selected/unselected
        # states, which makes the selected segment's text invisible against
        # its own selected_color background (same bug already fixed in the
        # handwritten_text mode selector and this app's kernel-preset
        # buttons) -- plain CTkButtons with per-state fg_color/text_color
        # avoid it entirely.
        self.transport_source_var = tk.StringVar(value="Processado")
        source_row = ctk.CTkFrame(card, fg_color="transparent")
        source_row.pack(fill="x", padx=14, pady=(0, 8))
        self._source_buttons = {}
        for label in ("Processado", "Original"):
            btn = ctk.CTkButton(source_row, text=label, corner_radius=8,
                                font=self.f_small, border_width=1, border_color=BORDER,
                                command=lambda l=label: self._select_transport_source(l))
            btn.pack(side="left", expand=True, fill="x", padx=3)
            self._source_buttons[label] = btn
        self._refresh_source_buttons()

        transport_row = ctk.CTkFrame(card, fg_color="transparent")
        transport_row.pack(fill="x", padx=14, pady=(0, 6))
        self.play_pause_btn = ctk.CTkButton(
            transport_row, text="Play", command=self._toggle_play_pause,
            fg_color=INK, hover_color=INK_HOVER, text_color=PAPER,
            font=self.f_body, width=90)
        self.play_pause_btn.pack(side="left", padx=(0, 6))
        ctk.CTkButton(transport_row, text="Stop", command=self._stop_playback,
                     fg_color=CARD, hover_color=BORDER, text_color=INK,
                     border_width=1, border_color=BORDER, font=self.f_body,
                     width=90).pack(side="left", padx=6)

        seek_row = ctk.CTkFrame(card, fg_color="transparent")
        seek_row.pack(fill="x", padx=14, pady=(2, 2))
        self.seek_slider = ctk.CTkSlider(
            seek_row, from_=0, to=1, number_of_steps=1000,
            command=self._on_seek_drag,
            fg_color=BORDER, progress_color=INK,
            button_color=INK, button_hover_color=INK_HOVER)
        self.seek_slider.set(0)
        self.seek_slider.bind("<Button-1>", self._on_scrub_press, add="+")
        self.seek_slider.bind("<ButtonRelease-1>", self._on_scrub_release, add="+")
        self.seek_slider.pack(fill="x")

        self.time_label_var = tk.StringVar(value="00:00 / 00:00")
        ctk.CTkLabel(card, textvariable=self.time_label_var, font=self.f_small,
                    text_color=MUTED).pack(anchor="e", padx=14, pady=(0, 8))

        ctk.CTkButton(card, text="Salvar processado como...", command=self._save_as,
                     fg_color=CARD, hover_color=BORDER, text_color=INK,
                     border_width=1, border_color=BORDER, font=self.f_body
                     ).pack(anchor="w", padx=14, pady=(4, 14))

    def _build_main(self, parent):
        card = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=14,
                            border_width=1, border_color=BORDER)
        card.grid(row=0, column=1, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)

        top = ctk.CTkFrame(card, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))
        ctk.CTkLabel(top, text="Analise", font=self.f_section, text_color=INK
                    ).pack(side="left")
        ctk.CTkButton(top, text="Atualizar", command=self._plot_current_view,
                     fg_color=INK, hover_color=INK_HOVER, text_color=PAPER,
                     font=self.f_body).pack(side="right")

        # Espectro (whole-signal FFT magnitude) vs Espectrograma (STFT,
        # magnitude over time AND frequency) -- plain buttons, not
        # CTkSegmentedButton, for the same invisible-selected-text reason as
        # the Processado/Original transport selector below.
        mode_row = ctk.CTkFrame(card, fg_color="transparent")
        mode_row.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
        self.view_mode_var = tk.StringVar(value="Espectro")
        self._view_mode_buttons = {}
        for label in ("Espectro", "Espectrograma", "Resposta EQ"):
            btn = ctk.CTkButton(mode_row, text=label, corner_radius=8,
                                font=self.f_small, border_width=1, border_color=BORDER,
                                command=lambda l=label: self._select_view_mode(l))
            btn.pack(side="left", expand=True, fill="x", padx=3)
            self._view_mode_buttons[label] = btn
        self._refresh_view_mode_buttons()

        self.spectrum_holder = ctk.CTkFrame(card, fg_color=PAPER, corner_radius=10)
        self.spectrum_holder.grid(row=2, column=0, sticky="nsew", padx=14, pady=(0, 14))
        card.grid_rowconfigure(2, weight=1)
        self.spectrum_canvas = None
        self._spectrum_placeholder = ctk.CTkLabel(
            self.spectrum_holder,
            text="Carregue um audio para visualizar\n(atualiza automaticamente a cada mudanca)",
            font=self.f_small, text_color=MUTED)
        self._spectrum_placeholder.pack(expand=True, fill="both", padx=20, pady=20)

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

    # ---- file actions -----------------------------------------------------
    def _load_audio(self):
        path = filedialog.askopenfilename(
            title="Carregar audio",
            filetypes=[("Audio", "*.wav *.flac *.ogg *.mp3"), ("Todos os arquivos", "*.*")])
        if not path:
            return
        try:
            audio, sr = audio_io.load_audio(path)
        except RuntimeError as exc:
            messagebox.showerror("Erro ao carregar audio", str(exc))
            return

        self.audio_path = path
        self._load_into_state(audio, sr, os.path.basename(path))

    def _load_into_state(self, audio, sr, label):
        """Seeds audio_original/audio, refreshes the file card, (re)wires
        the transport to the freshly loaded buffer, draws the spectrum and
        starts the position-poll loop. Split out from _load_audio so tests
        can preload a synthetic buffer without a real file dialog."""
        self.audio_original = audio.copy()
        self.audio = audio.copy()
        self.sr = sr

        self.file_label_var.set(label)
        duration = len(audio) / sr if sr else 0
        self.file_info_var.set(f"{duration:.2f}s @ {sr} Hz")
        self.status_var.set("Audio carregado.")

        self._load_transport_buffer()
        self._plot_current_view()
        self._start_tick()

    def _require_audio(self):
        if self.audio is None or self.sr is None:
            messagebox.showwarning("Aviso", "Carregue um arquivo de audio primeiro.")
            return False
        return True

    def _reset_to_original(self):
        if self.audio_original is None:
            messagebox.showwarning("Aviso", "Nenhum audio original carregado.")
            return

        # cancel any pending/in-flight recompute so a stale background
        # result can't land on top of the reset we're doing right now
        if self._recompute_job is not None:
            self.after_cancel(self._recompute_job)
            self._recompute_job = None
        self._recompute_gen += 1

        self.trim_on_var.set(False)
        self.compress_on_var.set(False)
        self.echo_on_var.set(False)
        self.reverb_on_var.set(False)
        self.eq_low_on_var.set(False)
        self.eq_mid_on_var.set(False)
        self.eq_high_on_var.set(False)

        for var, default, slider, lbl, fmt in (
            (self.trim_var, 10.0, self.trim_slider, self.trim_value_lbl, "{:.0f}s"),
            (self.compress_var, 0.5, self.compress_slider, self.compress_value_lbl, "{:.2f}"),
            (self.echo_delay_var, 0.5, self.echo_delay_slider, self.echo_delay_value_lbl, "{:.2f}"),
            (self.echo_gain_var, 0.6, self.echo_gain_slider, self.echo_gain_value_lbl, "{:.2f}"),
            (self.reverb_delays_var, 10, self.reverb_delays_slider, self.reverb_delays_value_lbl, "{:.0f}"),
            (self.reverb_time_var, 0.05, self.reverb_time_slider, self.reverb_time_value_lbl, "{:.2f}"),
            (self.eq_low_freq_var, 200.0, self.eq_low_freq_slider, self.eq_low_freq_value_lbl, "{:.0f}"),
            (self.eq_low_gain_var, 6.0, self.eq_low_gain_slider, self.eq_low_gain_value_lbl, "{:+.1f}"),
            (self.eq_mid_freq_var, 1000.0, self.eq_mid_freq_slider, self.eq_mid_freq_value_lbl, "{:.0f}"),
            (self.eq_mid_gain_var, -6.0, self.eq_mid_gain_slider, self.eq_mid_gain_value_lbl, "{:+.1f}"),
            (self.eq_high_freq_var, 5000.0, self.eq_high_freq_slider, self.eq_high_freq_value_lbl, "{:.0f}"),
            (self.eq_high_gain_var, 6.0, self.eq_high_gain_slider, self.eq_high_gain_value_lbl, "{:+.1f}"),
        ):
            var.set(default)
            slider.set(default)
            lbl.configure(text=fmt.format(default))

        self.audio = self.audio_original.copy()
        self._busy = False
        self.progress.set(0)
        self.status_var.set("Audio processado restaurado para o original.")

        if self._edit_active:
            self._cancel_spectral_edit(silent=True)
        self._load_transport_buffer()
        self._plot_current_view()

    # ---- effects (debounced auto-apply, background thread) -----------------
    def _schedule_recompute(self):
        """Called on every slider/switch change. Debounces ~300ms so a drag
        doesn't spawn a recompute per pixel -- only the settled-on value
        triggers work."""
        if self.audio_original is None:
            return
        if self._recompute_job is not None:
            self.after_cancel(self._recompute_job)
        self._recompute_job = self.after(300, self._recompute)

    def _recompute(self):
        self._recompute_job = None
        self._recompute_gen += 1
        gen = self._recompute_gen

        audio = self.audio_original.copy()
        sr = self.sr
        enabled = []
        if self.trim_on_var.get():
            enabled.append(("trim", dict(duration=self.trim_var.get())))
        if self.compress_on_var.get():
            enabled.append(("compress", dict(p=self.compress_var.get())))
        if self.echo_on_var.get():
            enabled.append(("echo", dict(delay=self.echo_delay_var.get(),
                                         echo_gain=self.echo_gain_var.get())))
        if self.reverb_on_var.get():
            enabled.append(("reverb", dict(num_delays=int(self.reverb_delays_var.get()),
                                           delay_time=self.reverb_time_var.get())))
        eq_bands = self._current_eq_bands()
        if eq_bands:
            enabled.append(("eq", dict(bands=eq_bands)))

        if not enabled:
            # nothing to do -- no need for a background thread, and this
            # keeps "all switches off" snappy/synchronous
            self._recompute_done(gen, audio)
            return

        self._busy = True
        self.progress.set(0.15)
        self.status_var.set("Recalculando efeitos...")
        threading.Thread(target=self._run_recompute, args=(gen, audio, sr, enabled),
                        daemon=True).start()

    def _run_recompute(self, gen, audio, sr, enabled):
        """Runs in a background thread -- effects like reverb on long clips
        can take a while, so keep the window responsive. Widgets are only
        ever touched back on the main thread, via self.after(...)."""
        try:
            result = audio
            for kind, params in enabled:
                if kind == "trim":
                    result = effects.trim_audio(result, sr, duration=params["duration"])
                elif kind == "compress":
                    result = effects.compress_audio(result, params["p"])
                elif kind == "echo":
                    result = effects.add_echo(result, sr, delay=params["delay"],
                                              echo_gain=params["echo_gain"])
                elif kind == "reverb":
                    result = effects.add_reverb(result, sr, num_delays=params["num_delays"],
                                                delay_time=params["delay_time"])
                elif kind == "eq":
                    result = filters_mod.apply_eq_bands(result, sr, params["bands"])
        except Exception as exc:
            self.after(0, self._recompute_failed, gen, str(exc))
            return
        self.after(0, self._recompute_done, gen, result)

    def _recompute_done(self, gen, result):
        if gen != self._recompute_gen:
            return  # a newer recompute superseded this one -- discard
        self.audio = result
        if self._edit_active:
            # A spectral-edit session snapshots the STFT of a specific
            # version of self.audio; if an effect change just replaced that
            # audio out from under it, the in-progress paint session no
            # longer corresponds to anything real -- drop it rather than let
            # "Aplicar" silently bake a stale edit onto the new audio.
            self._cancel_spectral_edit(silent=True)
        self._busy = False
        self.progress.set(1.0)
        self.status_var.set("Efeitos atualizados.")
        self._load_transport_buffer()
        self._plot_current_view()

    def _recompute_failed(self, gen, msg):
        if gen != self._recompute_gen:
            return
        self._busy = False
        self.progress.set(0)
        self.status_var.set("Erro.")
        messagebox.showerror("Erro ao recalcular efeitos", msg)

    # ---- spectrum / spectrogram -----------------------------------------
    def _select_view_mode(self, label):
        if self._edit_active and label != "Espectrograma":
            messagebox.showwarning(
                "Edicao em andamento",
                "Aplique ou cancele a edicao espectral antes de trocar de visualizacao.")
            return
        self.view_mode_var.set(label)
        self._refresh_view_mode_buttons()
        self._plot_current_view()

    def _refresh_view_mode_buttons(self):
        current = self.view_mode_var.get()
        for label, btn in self._view_mode_buttons.items():
            if label == current:
                btn.configure(fg_color=INK, hover_color=INK_HOVER, text_color=PAPER)
            else:
                btn.configure(fg_color=CARD, hover_color=BORDER, text_color=INK)

    def _on_spectrogram_settings_change(self):
        """n_fft/hop/window only affect how the CURRENT audio is drawn, not
        the audio itself -- redraw immediately if the spectrogram view is the
        one on screen, skip the work otherwise. These controls are disabled
        while a spectral-edit session is active (see _set_spectral_controls_
        enabled), so this should not normally fire mid-edit, but the guard
        stays as a defensive no-op in case it ever does."""
        if self._edit_active:
            return
        if self.view_mode_var.get() == "Espectrograma":
            self._plot_current_view()

    def _plot_current_view(self):
        mode = self.view_mode_var.get()
        if mode == "Espectrograma":
            self._plot_spectrogram()
        elif mode == "Resposta EQ":
            self._plot_eq_response()
        else:
            self._plot_magnitude_spectrum()

    def _plot_eq_response(self):
        """Analytic combined frequency response of the currently-enabled EQ
        bands -- computed directly from the filter coefficients, no audio
        needed, so it updates instantly as you move a slider (unlike the
        spectrogram, which needs the actual processed audio)."""
        if self.sr is None:
            self._require_audio()
            return
        bands = self._current_eq_bands()
        self._clear_plot_area()
        fig = Figure(figsize=(6, 3.6), dpi=100)
        ax = fig.add_subplot(111)
        if not bands:
            ax.text(0.5, 0.5, "Nenhuma banda de EQ ativa",
                   ha="center", va="center", color=MUTED, transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            total_db = None
            w = None
            for band in bands:
                b, a = filters_mod.build_biquad(band["type"], band["freq"], self.sr,
                                                q=band["q"], gain_db=band["gain_db"])
                w, mag_db = filters_mod.frequency_response(b, a, self.sr, n_points=1024)
                total_db = mag_db if total_db is None else total_db + mag_db
            ax.semilogx(w, total_db, color=INK, linewidth=1.2)
            ax.set_xlabel("Frequencia (Hz)")
            ax.set_ylabel("Ganho (dB)")
            ax.set_title("Resposta em frequencia do EQ (analitica)")
            ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()

        self.spectrum_canvas = FigureCanvasTkAgg(fig, master=self.spectrum_holder)
        self.spectrum_canvas.draw()
        self.spectrum_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.status_var.set("Resposta do EQ atualizada.")

    def _clear_plot_area(self):
        for widget in self.spectrum_holder.winfo_children():
            widget.destroy()

    def _plot_magnitude_spectrum(self):
        if not self._require_audio():
            return
        freqs, mags = spectrum_mod.magnitude_spectrum(self.audio, self.sr)

        self._clear_plot_area()
        fig = Figure(figsize=(6, 3.6), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(freqs, mags, color=INK, linewidth=0.8)
        ax.set_xlabel("Frequencia (Hz)")
        ax.set_title("Espectro de magnitude (visao geral, sem eixo do tempo)")
        fig.tight_layout()

        self.spectrum_canvas = FigureCanvasTkAgg(fig, master=self.spectrum_holder)
        self.spectrum_canvas.draw()
        self.spectrum_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.status_var.set("Espectro atualizado.")

    def _current_stft_settings(self):
        n_fft = int(self.stft_nfft_var.get())
        overlap_frac = self.stft_overlap_var.get() / 100.0
        hop = max(1, int(round(n_fft * (1.0 - overlap_frac))))
        window = self.stft_window_var.get()
        return n_fft, hop, overlap_frac, window

    def _plot_spectrogram(self):
        if not self._require_audio():
            return
        n_fft, hop, overlap_frac, window = self._current_stft_settings()

        try:
            times, freqs, db = stft_mod.spectrogram_db(self.audio, self.sr,
                                                       n_fft=n_fft, hop=hop, window=window)
        except Exception as exc:
            messagebox.showerror("Erro ao calcular espectrograma", str(exc))
            return

        self._clear_plot_area()
        fig = Figure(figsize=(6, 3.6), dpi=100)
        ax = fig.add_subplot(111)
        mesh = ax.pcolormesh(times, freqs, db, shading="gouraud", cmap="magma",
                            vmin=-80, vmax=0)
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Frequencia (Hz)")
        ax.set_title(f"Espectrograma -- janela {n_fft} amostras, "
                     f"{overlap_frac*100:.0f}% sobreposicao, {window}")
        fig.colorbar(mesh, ax=ax, label="dB")
        fig.tight_layout()

        self.spectrum_canvas = FigureCanvasTkAgg(fig, master=self.spectrum_holder)
        self.spectrum_canvas.draw()
        self.spectrum_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.status_var.set("Espectrograma atualizado.")

    # ---- spectral editing (paint/erase on the spectrogram, ISTFT back) -----
    def _select_paint_mode(self, label):
        self.paint_mode_var.set(label)
        self._refresh_paint_mode_buttons()

    def _refresh_paint_mode_buttons(self):
        current = self.paint_mode_var.get()
        for label, btn in self._paint_mode_buttons.items():
            if label == current:
                btn.configure(fg_color=INK, hover_color=INK_HOVER, text_color=PAPER)
            else:
                btn.configure(fg_color=CARD, hover_color=BORDER, text_color=INK)

    def _set_spectral_controls_enabled(self, enabled):
        """The STFT settings (window size/overlap/window function) define the
        shape of an in-progress edit's mask -- changing them mid-edit would
        invalidate it, so they're locked while editing."""
        state = "normal" if enabled else "disabled"
        self.stft_nfft_slider.configure(state=state)
        self.stft_overlap_slider.configure(state=state)
        self.stft_window_menu.configure(state=state)

    def _start_spectral_edit(self):
        if not self._require_audio():
            return
        if self.view_mode_var.get() != "Espectrograma":
            self.view_mode_var.set("Espectrograma")
            self._refresh_view_mode_buttons()

        n_fft, hop, _overlap_frac, window = self._current_stft_settings()
        try:
            freqs, times, S = stft_mod.stft(self.audio, self.sr, n_fft=n_fft, hop=hop, window=window)
        except Exception as exc:
            messagebox.showerror("Erro ao iniciar edicao espectral", str(exc))
            return

        self._edit_freqs, self._edit_times, self._edit_S = freqs, times, S
        self._edit_hop = hop
        self._edit_mask = np.ones(S.shape, dtype=np.float64)
        self._edit_active = True
        self._set_spectral_controls_enabled(False)
        self._render_spectral_edit_canvas()
        self.status_var.set(
            "Edicao espectral iniciada -- arraste no espectrograma para pintar "
            f"({self.paint_mode_var.get()}).")

    def _edit_db(self):
        mag = np.abs(self._edit_S * self._edit_mask)
        peak = np.abs(self._edit_S).max()
        ref = peak if peak > 0 else 1.0
        db = 20 * np.log10(np.maximum(mag, 1e-10) / ref)
        return np.maximum(db, -80.0)

    def _render_spectral_edit_canvas(self):
        self._clear_plot_area()
        db = self._edit_db()
        fig = Figure(figsize=(6, 3.6), dpi=100)
        ax = fig.add_subplot(111)
        mesh = ax.pcolormesh(self._edit_times, self._edit_freqs, db, shading="gouraud",
                             cmap="magma", vmin=-80, vmax=0)
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Frequencia (Hz)")
        ax.set_title("Edicao espectral -- arraste para pintar")
        fig.colorbar(mesh, ax=ax, label="dB")
        fig.tight_layout()

        self._edit_mesh = mesh
        self.spectrum_canvas = FigureCanvasTkAgg(fig, master=self.spectrum_holder)
        self.spectrum_canvas.draw()
        self.spectrum_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.spectrum_canvas.mpl_connect("button_press_event", self._on_canvas_press)
        self.spectrum_canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self.spectrum_canvas.mpl_connect("button_release_event", self._on_canvas_release)

    def _on_canvas_press(self, event):
        if not self._edit_active or event.inaxes is None:
            return
        self._painting = True
        self._paint_at(event.xdata, event.ydata)

    def _on_canvas_motion(self, event):
        if not self._edit_active or not self._painting or event.inaxes is None:
            return
        self._paint_at(event.xdata, event.ydata)

    def _on_canvas_release(self, _event=None):
        self._painting = False

    def _paint_at(self, t_center, f_center):
        if t_center is None or f_center is None:
            return
        half_t = (self.brush_time_var.get() / 1000.0) / 2.0
        half_f = self.brush_freq_var.get() / 2.0
        gain = 0.0 if self.paint_mode_var.get() == "Apagar" else 2.5
        stft_mod.paint_region(self._edit_mask, self._edit_freqs, self._edit_times,
                              (max(0.0, f_center - half_f), f_center + half_f),
                              (t_center - half_t, t_center + half_t), gain)
        self._edit_mesh.set_array(self._edit_db().ravel())
        self.spectrum_canvas.draw_idle()

    def _clear_spectral_mask(self):
        if not self._edit_active:
            return
        self._edit_mask[:] = 1.0
        self._edit_mesh.set_array(self._edit_db().ravel())
        self.spectrum_canvas.draw_idle()
        self.status_var.set("Mascara de edicao limpa.")

    def _cancel_spectral_edit(self, silent=False):
        if not self._edit_active:
            return
        self._edit_active = False
        self._edit_S = None
        self._edit_mask = None
        self._edit_mesh = None
        self._painting = False
        self._set_spectral_controls_enabled(True)
        self._plot_current_view()
        if not silent:
            self.status_var.set("Edicao espectral cancelada -- audio inalterado.")

    def _apply_spectral_edit(self):
        if not self._edit_active:
            return
        S_masked = self._edit_S * self._edit_mask
        try:
            new_audio = stft_mod.istft(S_masked, self.sr, hop=self._edit_hop,
                                       window=self.stft_window_var.get(),
                                       length=len(self.audio))
        except Exception as exc:
            messagebox.showerror("Erro ao aplicar edicao espectral", str(exc))
            return

        self.audio = new_audio
        self._edit_active = False
        self._edit_S = None
        self._edit_mask = None
        self._edit_mesh = None
        self._painting = False
        self._set_spectral_controls_enabled(True)
        self._load_transport_buffer()
        self._plot_current_view()
        self.status_var.set("Edicao espectral aplicada -- audio atualizado.")

    # ---- playback / transport ----------------------------------------------
    def _current_transport_buffer(self):
        if self.transport_source_var.get() == "Original":
            return self.audio_original
        return self.audio

    def _load_transport_buffer(self):
        """(Re)loads whichever buffer the Original/Processado selector
        points at into the player. Called after loading a file, after every
        recompute, and when the user flips the selector. Player.load()
        always rewinds to 0, so if audio was playing (or paused mid-way)
        before the swap, we restore that same position -- and keep it
        playing -- on the new buffer instead of yanking playback back to
        the start every time a slider tweaks the processed audio."""
        buf = self._current_transport_buffer()
        if buf is None or self.sr is None:
            return
        was_playing = self.player.is_playing()
        pos = self.player.position_seconds()

        self.player.load(buf, self.sr)
        duration = self.player.duration_seconds()
        self.seek_slider.configure(from_=0, to=max(duration, 0.001))

        pos = min(pos, duration)
        self.player.seek(pos)
        if was_playing and duration > 0:
            self.player.play()

        self.seek_slider.set(pos)
        self._sync_play_pause_button()
        self._update_time_label()

    def _select_transport_source(self, label):
        self.transport_source_var.set(label)
        self._refresh_source_buttons()
        self._load_transport_buffer()

    def _refresh_source_buttons(self):
        current = self.transport_source_var.get()
        for label, btn in self._source_buttons.items():
            if label == current:
                btn.configure(fg_color=INK, hover_color=INK_HOVER, text_color=PAPER)
            else:
                btn.configure(fg_color=CARD, hover_color=BORDER, text_color=INK)

    def _sync_play_pause_button(self):
        self.play_pause_btn.configure(text="Pause" if self.player.is_playing() else "Play")

    def _toggle_play_pause(self):
        if not self._require_audio():
            return
        if self.player.is_playing():
            self.player.pause()
            self.status_var.set("Reproducao pausada.")
        else:
            self.player.play()
            source = "original" if self.transport_source_var.get() == "Original" else "processado"
            self.status_var.set(f"Reproduzindo audio {source}...")
        self._sync_play_pause_button()
        self._start_tick()

    def _stop_playback(self):
        self.player.stop()
        self._sync_play_pause_button()
        self.seek_slider.set(0)
        self._update_time_label()
        self.status_var.set("Reproducao interrompida.")

    def _on_seek_drag(self, value):
        if self.player.duration_seconds() <= 0:
            return
        self.player.seek(float(value))
        self._update_time_label()

    def _on_scrub_press(self, _event=None):
        self._scrub_dragging = True

    def _on_scrub_release(self, _event=None):
        self._scrub_dragging = False
        self._on_seek_drag(self.seek_slider.get())

    def _update_time_label(self):
        def fmt(t):
            t = max(0, int(t))
            return f"{t // 60:02d}:{t % 60:02d}"
        self.time_label_var.set(
            f"{fmt(self.player.position_seconds())} / {fmt(self.player.duration_seconds())}")

    def _start_tick(self):
        if self._tick_job is None:
            self._tick_job = self.after(200, self._tick)

    def _tick(self):
        if self.player.finished():
            self.player.stop()
            self._sync_play_pause_button()

        if not self._scrub_dragging:
            self.seek_slider.set(self.player.position_seconds())
            self._update_time_label()

        self._tick_job = self.after(200, self._tick)

    def _save_as(self):
        if not self._require_audio():
            return
        path = filedialog.asksaveasfilename(
            title="Salvar audio processado", defaultextension=".wav",
            filetypes=[("WAV", "*.wav")])
        if not path:
            return
        audio_io.save_audio(self.audio, self.sr, path)
        self.status_var.set(f"Salvo em {path}")

    # ---- lifecycle -----------------------------------------------------
    def _on_close(self):
        if self._busy and not messagebox.askyesno(
                "Processando...", "Um efeito esta sendo aplicado. Fechar mesmo assim?"):
            return
        if self._tick_job is not None:
            self.after_cancel(self._tick_job)
            self._tick_job = None
        if self._recompute_job is not None:
            self.after_cancel(self._recompute_job)
            self._recompute_job = None
        self.player.stop()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
