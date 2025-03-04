import tkinter as tk
from tkinter import filedialog, ttk
import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
##################################################################################################

# =============================================================================
# audo procss functn
# =============================================================================
# =============================================================================
# funcs de procss de audo
# =============================================================================
############################################################################

def trim_audio(audio, sr, duration=10):
    """corta o audo para a durco espcfc (em segnds)."""
    num_samples = int(duration * sr)
    return audio[:num_samples]
#####################################################################

def plot_audio_spectrum(audio, sr):
    """cria e retrna um objto figre com o espctr de magntd do audo."""
    ft = fft(audio)
    magnitude_spectrum = np.abs(ft)
    frequencies = np.linspace(0, sr, len(magnitude_spectrum))
##############################################################################################
    
    fig = Figure(figsize=(8, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.plot(frequencies, magnitude_spectrum)
    ax.set_xlabel("Frequência (Hz)")
    ax.set_title("Espectro de Magnitude")
    return fig
###########################################################################

def compress_audio(audio, p):
    """presrv apns uma frco p das freqnc mais baxs."""
    ft = fft(audio)
    limit_index = int(len(ft) * p)
    ft_filtered = np.zeros_like(ft)
    ft_filtered[:limit_index] = ft[:limit_index]
    ft_filtered[-limit_index:] = ft[-limit_index:]
    return ifft(ft_filtered).real
#####################################################################

def add_echo(audio, sr, delay=0.5, echo_gain=0.6):
    """adicn um eco com o atrso espcfc (em segnds) e um ganho para o eco."""
    delay_samples = int(sr * delay)
    echo_signal = np.zeros_like(audio)
    echo_signal[delay_samples:] = audio[:-delay_samples]
    return audio + echo_gain * echo_signal
#####################################################################################

def add_reverb(audio, sr, num_delays=10, delay_time=0.05):
    """adicn multpl ecos (revrb) sem wrap-arnd, utilzn zero-paddng."""
    delay_samples = int(sr * delay_time)
    reverb_audio = audio.copy()
    for i in range(1, num_delays + 1):
        padded = np.concatenate((np.zeros(i * delay_samples), audio[:-i * delay_samples]))
        reverb_audio += padded / (i + 1)
    return reverb_audio
#########################################################################################

def save_audio(audio, sr, filename="processed_audio.wav"):
    """salva o audo procss no arqvo espcfc."""
    sf.write(filename, audio, sr)
    print(f"Áudio salvo como {filename}")
#################################################################################################

def play_audio_file(filename="processed_audio.wav"):
    """opns the procss audo file in the deflt extrnl plyr."""
    try:
        os.startfile(filename)
    except Exception as e:
        print("Error playing audio:", e)
####################################################################################

def play_audio_array(audio, sr, num_channels=1):
    """plys audo from a numpy arry usng the deflt extrnl plyr by savng it temprr."""
    temp_filename = "temp_audio.wav"
    save_audio(audio, sr, filename=temp_filename)
    try:
        os.startfile(temp_filename)
    except Exception as e:
        print("Error playing audio:", e)
##################################################################################################

# =============================================================================
# graphc intrfc (tkntr)
# =============================================================================
####################################################################

class AudioProcessingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio Processing Interface")
        self.geometry("1000x600")
##########################################################################################

        self.audio = None           # procss audo
        self.audio_original = None  # orignl audo copy
        self.sr = None
        self.audio_path = None
###############################################################################

        # side panl
        self.left_frame = tk.Frame(self, width=200, bg="#e0e0e0")
        self.left_frame.pack(side="left", fill="y")
###############################################################################################

        # main disply area
        self.right_frame = tk.Frame(self, bg="white")
        self.right_frame.pack(side="right", fill="both", expand=True)
#########################################################################################

        # dictnr for effct panls
        self.panels = {}
        self.create_panels()
        self.create_menu_buttons()
#################################################################

        # show intl panl
        self.show_panel("load")
###########################################################################

    def create_menu_buttons(self):
        """crts the menu buttns for each effct."""
        buttons = [
            ("Load Audio", "load"),
            ("Trim Audio", "trim"),
            ("Fourier Spectrum", "spectrum"),
            ("Fourier Compression", "compress"),
            ("Add Echo", "echo"),
            ("Add Reverb", "reverb"),
        ]
        for text, key in buttons:
            btn = tk.Button(self.left_frame, text=text, command=lambda k=key: self.show_panel(k),
                            relief="raised", padx=10, pady=5)
            btn.pack(fill="x", padx=10, pady=5)
#####################################################################

    def create_panels(self):
        """crts the panls for each effct."""
        # panl for lodng audo
        panel_load = tk.Frame(self.right_frame, bg="white")
        btn_select = tk.Button(panel_load, text="Select Audio", command=self.load_audio)
        btn_select.pack(pady=20)
        btn_play_orig = tk.Button(panel_load, text="Play Original Audio", command=self.play_original_audio)
        btn_play_orig.pack(pady=10)
        self.panels["load"] = panel_load
##########################################################################

        # panl for trimmn audo
        panel_trim = tk.Frame(self.right_frame, bg="white")
        tk.Label(panel_trim, text="Trim Audio Duration (sec):", bg="white").pack(pady=5)
        self.trim_scale = tk.Scale(panel_trim, from_=1, to=30, orient="horizontal", bg="white")
        self.trim_scale.set(10)
        self.trim_scale.pack(pady=5)
        btn_trim = tk.Button(panel_trim, text="Apply Trim", command=self.trim_audio)
        btn_trim.pack(pady=10)
        btn_play_trim = tk.Button(panel_trim, text="Play Processed Audio", command=self.play_processed_audio)
        btn_play_trim.pack(pady=5)
        self.panels["trim"] = panel_trim
########################################################################################

        # panl for forr spectr
        panel_spectrum = tk.Frame(self.right_frame, bg="white")
        btn_spectrum = tk.Button(panel_spectrum, text="Plot Fourier Spectrum", command=self.plot_spectrum)
        btn_spectrum.pack(pady=10)
        self.spectrum_canvas = None  # will hold the embddd plot
        self.panels["spectrum"] = panel_spectrum
##################################################################

        # panl for forr comprs
        panel_compress = tk.Frame(self.right_frame, bg="white")
        tk.Label(panel_compress, text="Compression Level (0.1 - 1):", bg="white").pack(pady=5)
        self.compress_scale = tk.Scale(panel_compress, from_=0.1, to=1.0, resolution=0.1, orient="horizontal", bg="white")
        self.compress_scale.set(0.5)
        self.compress_scale.pack(pady=5)
        btn_compress = tk.Button(panel_compress, text="Apply Compression", command=self.compress_audio)
        btn_compress.pack(pady=10)
        btn_play_compress = tk.Button(panel_compress, text="Play Processed Audio", command=self.play_processed_audio)
        btn_play_compress.pack(pady=5)
        self.panels["compress"] = panel_compress
############################################################################################

        # panl for echo effct
        panel_echo = tk.Frame(self.right_frame, bg="white")
        tk.Label(panel_echo, text="Echo Delay (sec):", bg="white").pack(pady=5)
        self.echo_scale = tk.Scale(panel_echo, from_=0.1, to=2.0, resolution=0.1, orient="horizontal", bg="white")
        self.echo_scale.set(0.5)
        self.echo_scale.pack(pady=5)
        btn_echo = tk.Button(panel_echo, text="Apply Echo", command=self.apply_echo)
        btn_echo.pack(pady=10)
        btn_play_echo = tk.Button(panel_echo, text="Play Processed Audio", command=self.play_processed_audio)
        btn_play_echo.pack(pady=5)
        self.panels["echo"] = panel_echo
#################################################################

        # panl for revrb effct
        panel_reverb = tk.Frame(self.right_frame, bg="white")
        tk.Label(panel_reverb, text="Number of Delays:", bg="white").pack(pady=5)
        self.reverb_delays_scale = tk.Scale(panel_reverb, from_=1, to=20, orient="horizontal", bg="white")
        self.reverb_delays_scale.set(10)
        self.reverb_delays_scale.pack(pady=5)
        tk.Label(panel_reverb, text="Delay Time (sec):", bg="white").pack(pady=5)
        self.reverb_time_scale = tk.Scale(panel_reverb, from_=0.01, to=0.5, resolution=0.01, orient="horizontal", bg="white")
        self.reverb_time_scale.set(0.05)
        self.reverb_time_scale.pack(pady=5)
        btn_reverb = tk.Button(panel_reverb, text="Apply Reverb", command=self.apply_reverb)
        btn_reverb.pack(pady=10)
        btn_play_reverb = tk.Button(panel_reverb, text="Play Processed Audio", command=self.play_processed_audio)
        btn_play_reverb.pack(pady=5)
        self.panels["reverb"] = panel_reverb
#################################################################

        # plce all panls in the same area (only one visble at a time)
        for panel in self.panels.values():
            panel.place(relx=0, rely=0, relwidth=1, relheight=1)
#######################################################################

    def show_panel(self, key):
        """disply the selctd panl."""
        panel = self.panels.get(key)
        if panel:
            panel.tkraise()
###########################################################################################

    def load_audio(self):
        """lods an audo file usng soundf.read."""
        self.audio_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])
        if self.audio_path:
            audio, sr = sf.read(self.audio_path)
            # convrt to mono if necssr.
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            self.audio_original = audio.copy()
            self.audio = audio.copy()
            self.sr = sr
            print(f"Audio loaded: {self.audio_path}")
##############################################################################################

    def trim_audio(self):
        """trms the audo to the specfd durtn."""
        if self.audio is None:
            print("No audio loaded!")
            return
        duration = self.trim_scale.get()
        self.audio = trim_audio(self.audio, self.sr, duration=duration)
        save_audio(self.audio, self.sr)
####################################################################################

    def plot_spectrum(self):
        """embds the forr spectr plot into the spectr panl."""
        if self.audio is None:
            print("No audio loaded!")
            return
        fig = plot_audio_spectrum(self.audio, self.sr)
        # cler prevs canvs if exsts.
        for widget in self.panels["spectrum"].winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.panels["spectrum"])
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
#########################################################################

    def compress_audio(self):
        """appls forr comprs usng the specfd fractn."""
        if self.audio is None:
            print("No audio loaded!")
            return
        p = self.compress_scale.get()
        self.audio = compress_audio(self.audio, p)
        save_audio(self.audio, self.sr)
####################################################################################

    def apply_echo(self):
        """appls an echo effct usng the selctd dely."""
        if self.audio is None:
            print("No audio loaded!")
            return
        delay = self.echo_scale.get()
        self.audio = add_echo(self.audio, self.sr, delay=delay)
        save_audio(self.audio, self.sr)
##################################################################

    def apply_reverb(self):
        """appls a revrb effct usng the selctd parmtr."""
        if self.audio is None:
            print("No audio loaded!")
            return
        num_delays = self.reverb_delays_scale.get()
        delay_time = self.reverb_time_scale.get()
        self.audio = add_reverb(self.audio, self.sr, num_delays=num_delays, delay_time=delay_time)
        save_audio(self.audio, self.sr)
#######################################################################################

    def play_processed_audio(self):
        """opns the procss audo in the deflt extrnl plyr."""
        play_audio_file("processed_audio.wav")
#################################################################

    def play_original_audio(self):
        """opns the orignl audo file in the deflt extrnl plyr."""
        if self.audio_path is None:
            print("No original audio loaded!")
            return
        try:
            os.startfile(self.audio_path)
        except Exception as e:
            print("Error playing original audio:", e)
##################################################################

if __name__ == "__main__":
    app = AudioProcessingApp()
    app.mainloop()