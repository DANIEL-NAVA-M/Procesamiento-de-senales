import wave
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import tkinter as tk
from tkinter import filedialog, ttk
import soundfile as sf
import pyaudio

# Variables globales
audio_file = None
processed_signal = None
sample_rate = None

# Funciones para procesamiento de señal
def load_audio():
    global audio_file, sample_rate
    filepath = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if not filepath:
        return
    audio_file = filepath
    data, sample_rate = sf.read(audio_file)
    if len(data.shape) > 1:  # Convertir a mono si es estéreo
        data = data.mean(axis=1)
    plot_signal(data, "Señal Original")

def apply_filter():
    global processed_signal
    if audio_file is None:
        return
    
    cutoff = float(cutoff_slider.get())
    order = int(order_slider.get())
    filter_type = filter_var.get()
    
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    
    if filter_type == "Pasa-baja":
        b, a = signal.butter(order, normal_cutoff, btype='low')
    elif filter_type == "Pasa-alta":
        b, a = signal.butter(order, normal_cutoff, btype='high')
    elif filter_type == "Pasa-banda":
        high_cutoff = float(high_cutoff_slider.get()) / nyquist
        b, a = signal.butter(order, [normal_cutoff, high_cutoff], btype='band')
    else:
        return
    
    raw_data, _ = sf.read(audio_file)
    processed_signal = signal.filtfilt(b, a, raw_data)
    plot_signal(processed_signal, "Señal Filtrada")

def apply_fft():
    if audio_file is None:
        return
    data, _ = sf.read(audio_file)
    freq = np.fft.fftfreq(len(data), d=1/sample_rate)
    fft_signal = np.abs(np.fft.fft(data))
    plt.figure()
    plt.plot(freq[:len(freq)//2], fft_signal[:len(freq)//2])
    plt.title("Transformada de Fourier")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    plt.show()

def save_audio():
    if processed_signal is None:
        return
    save_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
    if save_path:
        sf.write(save_path, processed_signal, sample_rate)

def play_audio(original=True):
    if audio_file is None:
        return
    
    data, _ = sf.read(audio_file) if original else (processed_signal, sample_rate)
    
    if data is None:
        return
    
    audio = (data * 32767).astype(np.int16).tobytes()
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, output=True)
    stream.write(audio)
    stream.stop_stream()
    stream.close()
    p.terminate()

def plot_signal(data, title):
    plt.figure()
    plt.plot(np.linspace(0, len(data) / sample_rate, num=len(data)), data)
    plt.title(title)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.show()

# Configuración de la Interfaz
top = tk.Tk()
top.title("HMI para Procesamiento de Señales")
top.geometry("500x400")

tk.Button(top, text="Cargar Archivo", command=load_audio).pack()
tk.Button(top, text="Reproducir Original", command=lambda: play_audio(True)).pack()
tk.Button(top, text="Reproducir Procesado", command=lambda: play_audio(False)).pack()

tk.Label(top, text="Filtro:").pack()
filter_var = tk.StringVar(value="Pasa-baja")
ttkn = ttk.Combobox(top, textvariable=filter_var, values=["Pasa-baja", "Pasa-alta", "Pasa-banda"])
ttkn.pack()

tk.Label(top, text="Frecuencia de Corte (Hz):").pack()
cutoff_slider = tk.Scale(top, from_=100, to=10000, orient="horizontal")
cutoff_slider.pack()

tk.Label(top, text="Frecuencia Alta (Hz) para Pasa-banda:").pack()
high_cutoff_slider = tk.Scale(top, from_=500, to=15000, orient="horizontal")
high_cutoff_slider.pack()

tk.Label(top, text="Orden del Filtro:").pack()
order_slider = tk.Scale(top, from_=1, to=10, orient="horizontal")
order_slider.pack()

tk.Button(top, text="Aplicar Filtro", command=apply_filter).pack()
tk.Button(top, text="Aplicar Transformada", command=apply_fft).pack()
tk.Button(top, text="Guardar Resultado", command=save_audio).pack()

top.mainloop()
