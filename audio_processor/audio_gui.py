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

import customtkinter as ctk
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from audiodsp import io as audio_io
from audiodsp import effects
from audiodsp import spectrum as spectrum_mod
from audiodsp import playback

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
        card.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(card, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))
        ctk.CTkLabel(top, text="Espectro", font=self.f_section, text_color=INK
                    ).pack(side="left")
        ctk.CTkButton(top, text="Atualizar espectro", command=self._plot_spectrum,
                     fg_color=INK, hover_color=INK_HOVER, text_color=PAPER,
                     font=self.f_body).pack(side="right")

        self.spectrum_holder = ctk.CTkFrame(card, fg_color=PAPER, corner_radius=10)
        self.spectrum_holder.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        self.spectrum_canvas = None
        self._spectrum_placeholder = ctk.CTkLabel(
            self.spectrum_holder, text="Carregue um audio para visualizar o espectro"
            "\n(atualiza automaticamente a cada mudanca)",
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
        self._plot_spectrum()
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

        for var, default, slider, lbl, fmt in (
            (self.trim_var, 10.0, self.trim_slider, self.trim_value_lbl, "{:.0f}s"),
            (self.compress_var, 0.5, self.compress_slider, self.compress_value_lbl, "{:.2f}"),
            (self.echo_delay_var, 0.5, self.echo_delay_slider, self.echo_delay_value_lbl, "{:.2f}"),
            (self.echo_gain_var, 0.6, self.echo_gain_slider, self.echo_gain_value_lbl, "{:.2f}"),
            (self.reverb_delays_var, 10, self.reverb_delays_slider, self.reverb_delays_value_lbl, "{:.0f}"),
            (self.reverb_time_var, 0.05, self.reverb_time_slider, self.reverb_time_value_lbl, "{:.2f}"),
        ):
            var.set(default)
            slider.set(default)
            lbl.configure(text=fmt.format(default))

        self.audio = self.audio_original.copy()
        self._busy = False
        self.progress.set(0)
        self.status_var.set("Audio processado restaurado para o original.")

        self._load_transport_buffer()
        self._plot_spectrum()

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
        except Exception as exc:
            self.after(0, self._recompute_failed, gen, str(exc))
            return
        self.after(0, self._recompute_done, gen, result)

    def _recompute_done(self, gen, result):
        if gen != self._recompute_gen:
            return  # a newer recompute superseded this one -- discard
        self.audio = result
        self._busy = False
        self.progress.set(1.0)
        self.status_var.set("Efeitos atualizados.")
        self._load_transport_buffer()
        self._plot_spectrum()

    def _recompute_failed(self, gen, msg):
        if gen != self._recompute_gen:
            return
        self._busy = False
        self.progress.set(0)
        self.status_var.set("Erro.")
        messagebox.showerror("Erro ao recalcular efeitos", msg)

    # ---- spectrum -----------------------------------------------------
    def _plot_spectrum(self):
        if not self._require_audio():
            return
        freqs, mags = spectrum_mod.magnitude_spectrum(self.audio, self.sr)

        for widget in self.spectrum_holder.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(6, 3.6), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(freqs, mags, color=INK, linewidth=0.8)
        ax.set_xlabel("Frequencia (Hz)")
        ax.set_title("Espectro de magnitude")
        fig.tight_layout()

        self.spectrum_canvas = FigureCanvasTkAgg(fig, master=self.spectrum_holder)
        self.spectrum_canvas.draw()
        self.spectrum_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.status_var.set("Espectro atualizado.")

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
