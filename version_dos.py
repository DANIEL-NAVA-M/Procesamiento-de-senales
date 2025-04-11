# VERSION 2 (3 de abril del 2025)
# HMI para procesamiento de señales
# Daniel Nava Mondragón A0166161649


import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import wave
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QWidget,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QHBoxLayout
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


def lowpass_filter(signal_data, sample_rate, cutoff_freq, order):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return signal.filtfilt(b, a, signal_data)

def highpass_filter(signal_data, sample_rate, cutoff_freq, order):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return signal.filtfilt(b, a, signal_data)

def bandpass_filter(signal_data, sample_rate, low_cutoff, high_cutoff, order):
    nyquist = 0.5 * sample_rate
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = signal.butter(order, [low, high], btype='band', analog=False)
    return signal.filtfilt(b, a, signal_data)

def apply_fft(signal_data, sample_rate):
    n = len(signal_data)
    freq = np.fft.fftfreq(n, d=1/sample_rate)
    fft_signal = np.fft.fft(signal_data)
    return freq[:n//2], np.abs(fft_signal)[:n//2]

class SignalProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procesador de Señales de Audio")
        self.setGeometry(100, 100, 900, 700)

        self.audio_data = None
        self.sample_rate = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.load_button = QPushButton("Cargar archivo WAV")
        self.load_button.clicked.connect(self.load_audio)
        layout.addWidget(self.load_button)

        self.filter_box = QComboBox()
        self.filter_box.addItems(["Pasa baja", "Pasa alta", "Pasa banda", "FFT"])
        layout.addWidget(self.filter_box)

        freq_layout = QHBoxLayout()
        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(1, 20000)
        self.freq_spin.setValue(1000)
        self.freq_spin.setSuffix(" Hz")
        freq_layout.addWidget(QLabel("Frecuencia de corte 1:"))
        freq_layout.addWidget(self.freq_spin)

        self.freq_spin2 = QDoubleSpinBox()
        self.freq_spin2.setRange(1, 20000)
        self.freq_spin2.setValue(5000)
        self.freq_spin2.setSuffix(" Hz")
        freq_layout.addWidget(QLabel("Frecuencia de corte 2 (filtro pasa banda):"))
        freq_layout.addWidget(self.freq_spin2)
        layout.addLayout(freq_layout)

        self.order_spin = QSpinBox()
        self.order_spin.setRange(1, 10)
        self.order_spin.setValue(5)
        layout.addWidget(QLabel("Orden del filtro:"))
        layout.addWidget(self.order_spin)

        self.process_button = QPushButton("Aplicar filtro/transformada")
        self.process_button.clicked.connect(self.process_signal)
        layout.addWidget(self.process_button)

        self.figure, self.ax = plt.subplots(2, 1, figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo de audio", "", "WAV Files (*.wav)")
        if path:
            with wave.open(path, 'rb') as wav_file:
                self.sample_rate = wav_file.getframerate()
                frames = wav_file.readframes(-1)
                self.audio_data = np.frombuffer(frames, dtype=np.int16)
            self.plot_signal(self.audio_data, title="Señal Original")

    def process_signal(self):
        if self.audio_data is None:
            return

        signal_data = self.audio_data
        filter_type = self.filter_box.currentText()
        order = self.order_spin.value()
        cutoff1 = self.freq_spin.value()
        cutoff2 = self.freq_spin2.value()

        if filter_type == "Pasa baja":
            processed = lowpass_filter(signal_data, self.sample_rate, cutoff1, order)
        elif filter_type == "Pasa alta":
            processed = highpass_filter(signal_data, self.sample_rate, cutoff1, order)
        elif filter_type == "Pasa banda":
            processed = bandpass_filter(signal_data, self.sample_rate, cutoff1, cutoff2, order)
        elif filter_type == "FFT":
            freq, fft_data = apply_fft(signal_data, self.sample_rate)
            self.plot_fft(freq, fft_data)
            return

        self.plot_signal(processed, title="Señal Procesada")

    def plot_signal(self, data, title=""):
        time = np.linspace(0, len(data) / self.sample_rate, num=len(data))
        self.ax[0].cla()
        self.ax[1].cla()
        self.ax[0].plot(time, self.audio_data, label="Original", alpha=0.6)
        self.ax[1].plot(time, data, label=title, color='orange')
        self.ax[0].set_title("Señal Original")
        self.ax[1].set_title(title)
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_fft(self, freq, fft_data):
        self.ax[0].cla()
        self.ax[1].cla()
        self.ax[0].plot(self.audio_data[:len(freq)], label="Original")
        self.ax[0].set_title("Señal Original (dominio del tiempo)")
        self.ax[1].plot(freq, fft_data, color='purple')
        self.ax[1].set_title("Transformada de Fourier (FFT)")
        self.figure.tight_layout()
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SignalProcessor()
    window.show()
    sys.exit(app.exec_())
