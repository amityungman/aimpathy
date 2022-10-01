import pyaudio

# constants
CHUNK = 1024 * 2             # samples per frame
FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
RATE = 44100                 # samples per second
REFRESH_RATE = 0.03          # Rate of refresh of graphs in seconds
SMOOTH_RATE = 4              # Rate of smoothing of the display graph
GRAPH_SHOULDERS = 20         # The shoulders to buffer in the graph's display
