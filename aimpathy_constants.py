import pyaudio
import math
import numpy as np

# constants
CHUNK = 1024 * 8             # samples per frame
FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
RATE = 44100                 # samples per second
REFRESH_RATE = 0.005           # Rate of refresh of graphs in seconds
SMOOTH_RATE = 4              # Rate of smoothing of the display graph
GRAPH_SHOULDERS = 20         # The shoulders to buffer in the graph's display

FFT_POINTS = CHUNK // 2
FFT_POINTS_UNSIGNED = CHUNK // 4
SECS_PER_SPECTROGRAM_SEGMENT = (FFT_POINTS - (FFT_POINTS // 8)) / RATE
SPECTROGRAM_SECS = 5
SPECTROGRAM_MIN_VALUE = 2 ** 11
SPECTROGRAM_MAX_VALUE = 2 ** 15
SPECTROGRAM_MIN_FREQUENCY = 100
SPECTROGRAM_MAX_FREQUENCY = RATE // 2
SPECTRO_X_SIZE = SPECTROGRAM_SECS * math.ceil(1 / SECS_PER_SPECTROGRAM_SEGMENT)     # num of spectrograms to display

THAYER_SCATTER_BUFFER_SIZE = 10
thayer_scatter_colors = ['#000000'] + ['#' + 3 * hex(100 + 122 // (THAYER_SCATTER_BUFFER_SIZE - i))[2:] for i in range(1, THAYER_SCATTER_BUFFER_SIZE)]
thayer_scatter_sizes = [50.0] + [50.0/(i + 1) for i in range(1, THAYER_SCATTER_BUFFER_SIZE)]
