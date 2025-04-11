# VERSION 1.3 (31 de marzo del 2025)
# HMI para procesamiento de señales
# Daniel Nava Mondragón A0166161649

import wave
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import scipy.io.wavfile as wav

audio = 'videoplayback.wav'

def lowpass_filter(signal_data, sample_rate, cutoff_freq, order=5):
    """
    Aplica un filtro pasa baja Butterworth a una señal de audio.
    """
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_signal = signal.filtfilt(b, a, signal_data)
    return filtered_signal

def highpass_filter(signal_data, sample_rate, cutoff_freq, order=10):
    """
    Aplica un filtro pasa alta Butterworth a una señal de audio.
    """
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    filtered_signal = signal.filtfilt(b, a, signal_data)
    return filtered_signal

def bandpass_filter(signal_data, sample_rate, low_cutoff, high_cutoff, order=5):
    """
    Aplica un filtro pasa banda Butterworth a una señal de audio.
    """
    nyquist = 0.5 * sample_rate
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = signal.butter(order, [low, high], btype='band', analog=False)
    filtered_signal = signal.filtfilt(b, a, signal_data)
    return filtered_signal

def apply_fft(signal_data, sample_rate):
    """
    Aplica la Transformada de Fourier a la señal de audio.
    """
    n = len(signal_data)
    freq = np.fft.fftfreq(n, d=1/sample_rate)
    fft_signal = np.fft.fft(signal_data)
    return freq[:n//2], np.abs(fft_signal)[:n//2]

def main():
    # Cargar el archivo de audio
    raw = wave.open(audio)
    signal_data = raw.readframes(-1)
    signal_data = np.frombuffer(signal_data, dtype="int16")
    f_rate = raw.getframerate()
    
    # Aplicar filtros
    cutoff_low = 1000  # Frecuencia de corte baja en Hz
    cutoff_high = 10000  # Frecuencia de corte alta en Hz
    
    low_filtered_signal = lowpass_filter(signal_data, f_rate, cutoff_low)
    high_filtered_signal = highpass_filter(signal_data, f_rate, cutoff_high)
    band_filtered_signal = bandpass_filter(signal_data, f_rate, cutoff_low, cutoff_high)
    
    # Aplicar Transformada de Fourier
    freq, fft_original = apply_fft(signal_data, f_rate)
    
    # Generar eje de tiempo
    time = np.linspace(0, len(signal_data) / f_rate, num=len(signal_data))
    
    # Graficar las señales
    plt.figure(figsize=(10, 10))
    
    plt.subplot(5, 1, 1)
    plt.title("Señal Original - " + audio)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.plot(time, signal_data, label='Original', alpha=0.7)
    
    plt.subplot(5, 1, 2)
    plt.title("Señal con filtro pasa baja")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.plot(time, low_filtered_signal, label='Pasa baja', color='red', alpha=0.7)
    
    plt.subplot(5, 1, 3)
    plt.title("Señal con filtro pasa alta")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.plot(time, high_filtered_signal, label='Pasa alta', color='green', alpha=0.7)
    
    plt.subplot(5, 1, 4)
    plt.title("Señal con filtro pasa banda")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.plot(time, band_filtered_signal, label='Pasa banda', color='blue', alpha=0.7)
    
    plt.subplot(5, 1, 5)
    plt.title("Espectro de Fourier de la Señal Original")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    plt.plot(freq, fft_original, label='FFT', color='purple', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()
