import pyaudio

# constants
CHUNK = 1024 * 8             # samples per frame
FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
RATE = 44100                 # samples per second
REFRESH_RATE = 0.01           # Rate of refresh of graphs in seconds
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
