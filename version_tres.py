# VERSION 4 - Solo WAV
# Daniel Nava Mondragón A0166161649

import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import wave
from scipy import signal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QWidget,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QHBoxLayout
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.io.wavfile import write as write_wav

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

def compute_fft(signal_data, sample_rate):
    n = len(signal_data)
    freq = np.fft.fftfreq(n, d=1/sample_rate)
    fft_signal = np.fft.fft(signal_data)
    return freq[:n//2], np.abs(fft_signal)[:n//2]

class SignalProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procesador de Señales de Audio")
        self.setGeometry(100, 100, 1000, 800)

        self.audio_data = None
        self.sample_rate = None
        self.processed_data = None
        self.audio_path = None
        self.processed_path = None
        self.process_count = 0  # Contador para nombres únicos

        self.player = QMediaPlayer()

        self.init_ui()


    def init_ui(self):
        layout = QVBoxLayout()

        self.load_button = QPushButton("Cargar archivo WAV")
        self.load_button.clicked.connect(self.load_audio)
        layout.addWidget(self.load_button)

        self.filter_box = QComboBox()
        self.filter_box.addItems(["Pasa baja", "Pasa alta", "Pasa banda"])
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
        freq_layout.addWidget(QLabel("Frecuencia de corte 2:"))
        freq_layout.addWidget(self.freq_spin2)
        layout.addLayout(freq_layout)

        self.order_spin = QSpinBox()
        self.order_spin.setRange(1, 10)
        self.order_spin.setValue(5)
        layout.addWidget(QLabel("Orden del filtro:"))
        layout.addWidget(self.order_spin)

        self.process_button = QPushButton("Aplicar filtro")
        self.process_button.clicked.connect(self.process_signal)
        layout.addWidget(self.process_button)

        btn_layout = QHBoxLayout()
        self.play_original_btn = QPushButton("Reproducir Original")
        self.play_original_btn.clicked.connect(self.play_original_audio)
        btn_layout.addWidget(self.play_original_btn)

        self.play_processed_btn = QPushButton("Reproducir y Descargar Procesada")
        self.play_processed_btn.clicked.connect(self.play_processed_audio)
        btn_layout.addWidget(self.play_processed_btn)

        layout.addLayout(btn_layout)

        self.figure, self.ax = plt.subplots(2, 2, figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo WAV", "", "WAV Files (*.wav)")
        if path:
            self.audio_path = path
            with wave.open(path, 'rb') as wav_file:
                self.sample_rate = wav_file.getframerate()
                frames = wav_file.readframes(-1)
                self.audio_data = np.frombuffer(frames, dtype=np.int16)
            self.processed_data = None
            self.plot_all()

    def process_signal(self):
        if self.audio_data is None:
            return

        signal_data = self.audio_data
        filter_type = self.filter_box.currentText()
        order = self.order_spin.value()
        cutoff1 = self.freq_spin.value()
        cutoff2 = self.freq_spin2.value()

        if filter_type == "Pasa baja":
            self.processed_data = lowpass_filter(signal_data, self.sample_rate, cutoff1, order)
        elif filter_type == "Pasa alta":
            self.processed_data = highpass_filter(signal_data, self.sample_rate, cutoff1, order)
        elif filter_type == "Pasa banda":
            self.processed_data = bandpass_filter(signal_data, self.sample_rate, cutoff1, cutoff2, order)

        # Incrementar el contador y generar un nuevo nombre
        self.process_count += 1
        self.processed_path = f"processed_{self.process_count}.wav"

        write_wav(self.processed_path, self.sample_rate, self.processed_data.astype(np.int16))
        self.plot_all()

    def plot_all(self):
        for ax in self.ax.flatten():
            ax.clear()

        if self.audio_data is None:
            return

        time = np.linspace(0, len(self.audio_data) / self.sample_rate, num=len(self.audio_data))
        freq_orig, fft_orig = compute_fft(self.audio_data, self.sample_rate)

        self.ax[0][0].plot(time, self.audio_data, label="Original")
        self.ax[0][0].set_title("Señal Original")
        self.ax[0][0].set_xlabel("Tiempo [s]")

        self.ax[0][1].plot(freq_orig, fft_orig, color='green')
        self.ax[0][1].set_title("FFT de Señal Original")
        self.ax[0][1].set_xlabel("Frecuencia [Hz]")

        if self.processed_data is not None:
            time_proc = np.linspace(0, len(self.processed_data) / self.sample_rate, num=len(self.processed_data))
            freq_proc, fft_proc = compute_fft(self.processed_data, self.sample_rate)

            self.ax[1][0].plot(time_proc, self.processed_data, color='orange')
            self.ax[1][0].set_title("Señal Procesada")
            self.ax[1][0].set_xlabel("Tiempo [s]")

            self.ax[1][1].plot(freq_proc, fft_proc, color='red')
            self.ax[1][1].set_title("FFT de Señal Procesada")
            self.ax[1][1].set_xlabel("Frecuencia [Hz]")

        self.figure.tight_layout()
        self.canvas.draw()

    def play_original_audio(self):
        if self.audio_path:
            url = QUrl.fromLocalFile(os.path.abspath(self.audio_path))
            self.player.setMedia(QMediaContent(url))
            self.player.play()

    def play_processed_audio(self):
        if os.path.exists(self.processed_path):
            url = QUrl.fromLocalFile(os.path.abspath(self.processed_path))
            self.player.setMedia(QMediaContent(url))
            self.player.play()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SignalProcessor()
    window.show()
    sys.exit(app.exec_())
