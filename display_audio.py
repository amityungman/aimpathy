from aimpathy_constants import *
from audio_utils import smooth
import pyaudio
import struct
import numpy as np
import matplotlib.pyplot as plt
import time


"""********************   STARTING AUDIO STREAM   ********************"""

# pyaudio class instance
p = pyaudio.PyAudio()

# stream object to get data from microphone
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)


"""********************   ADJUST DISPLAY WINDOW   ********************"""

# create matplotlib figure and axes
# fig, axes = plt.subplots(2, 2, figsize=(20, 7), gridspec_kw={'width_ratios': [3, 1]})
# ((ax_wave, ax_specto), (ax_thayer, ax_emotion)) = axes
fig, axes = plt.subplot_mosaic("AAB;CDD", figsize=(20, 7))
ax_wave, ax_specto, ax_thayer, ax_emotion = axes.values()

"""***   WAVEFORM   ***"""
ax_wave.set_title('AUDIO WAVEFORM')
ax_wave.set_xlabel('Samples')
ax_wave.set_ylabel('Volume')
ax_wave.set_ylim(0 - GRAPH_SHOULDERS, 255 + GRAPH_SHOULDERS)
ax_wave.set_xlim(0, 2 * CHUNK)
plt.setp(ax_wave, xticks=[0, CHUNK, 2 * CHUNK], yticks=[0, 128, 255])

# variable for plotting
wave_x = np.arange(0, 2 * CHUNK, 2)
wave_line, = ax_wave.plot(wave_x, np.random.rand(CHUNK), '-', lw=1)

"""***   SPECTROGRAM   ***"""
ax_specto.set_title('SPECTROGRAM')
ax_specto.set_xlabel('Time')
ax_specto.set_ylabel('Hz')
# ax_specto.set_ylim(0 - GRAPH_SHOULDERS, 255 + GRAPH_SHOULDERS)
# ax_specto.set_xlim(0, 2 * CHUNK)
# plt.setp(ax_specto, xticks=[0, CHUNK, 2 * CHUNK], yticks=[0, 128, 255])

"""***   THAYER MODEL   ***"""
ax_thayer.set_title('Valance-Arousal graph (Thayer model)')
ax_thayer.set_xlabel('Valance')
ax_thayer.set_ylabel('Arousal')
# ax_specto.set_ylim(0 - GRAPH_SHOULDERS, 255 + GRAPH_SHOULDERS)
# ax_specto.set_xlim(0, 2 * CHUNK)
# plt.setp(ax_specto, xticks=[0, CHUNK, 2 * CHUNK], yticks=[0, 128, 255])

"""***   EMOTIONS   ***"""
ax_emotion.set_title('Perceived emotion graph')
ax_emotion.set_xlabel('Time')
ax_emotion.set_ylabel('Emotion level')
ax_emotion.set_ylim(0 - GRAPH_SHOULDERS//5, 100 + GRAPH_SHOULDERS//5)
# ax_specto.set_xlim(0, 2 * CHUNK)
# plt.setp(ax_specto, xticks=[0, CHUNK, 2 * CHUNK], yticks=[0, 128, 255])

print("Display ready")
plt.show(block=False)

while True:
    # binary data
    data = stream.read(CHUNK)

    # convert data to integers, for the wave form graph
    wave_data_int = struct.unpack(str(2 * CHUNK) + 'B', data)
    wave_data_np = np.array(wave_data_int, dtype='b')[::2] + 128
    wave_data_np = smooth(wave_data_np, SMOOTH_RATE)
    wave_line.set_ydata(wave_data_np)

    # update figure canvas
    try:
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(REFRESH_RATE)

    except:
        print('stream stopped')
        break

print('ended')
