from aimpathy_constants import *
from models.thayer_detector import ThayerRandom
from audio_utils import smooth
from typing import List, Dict
import pyaudio
import struct
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy import signal
from models.thayer_to_emotion import thayer_coordinates_to_emotion, AimpathyEmotion


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

fig, axes = plt.subplot_mosaic("ABC;DDC", figsize=(20, 7))
ax_wave, ax_thayer, ax_spectro, ax_emotion = axes.values()

"""***   WAVEFORM   ***"""
ax_wave.set_title('Audioform')
ax_wave.set_xlabel('Samples')
ax_wave.set_ylabel('Volume')
ax_wave.set_ylim(0 - GRAPH_SHOULDERS, 255 + GRAPH_SHOULDERS)
ax_wave.set_xlim(0, 2 * CHUNK)
plt.setp(ax_wave, xticks=[0, CHUNK, 2 * CHUNK], yticks=[0, 128, 255])

wave_x = np.arange(0, 2 * CHUNK, 2)
wave_line, = ax_wave.plot(wave_x, np.random.rand(CHUNK), '-', lw=1)


"""***   SPECTROGRAM   ***"""
ax_spectro.set_title('Spectrogram')
ax_spectro.set_xlabel('Time [sec]')
ax_spectro.set_ylabel('Frequency [Hz]')
spectro_x_size = SPECTROGRAM_SECS * math.ceil(1 / SECS_PER_SPECTROGRAM_SEGMENT)  # num of spectrograms to display
spectro_y_size = FFT_POINTS_UNSIGNED + 1  # num of frequencies
spectro_x = np.array([(FFT_POINTS_UNSIGNED / RATE) + i * SECS_PER_SPECTROGRAM_SEGMENT for i in range(0, spectro_x_size)])
spectro_y = np.linspace(0, RATE // 2, spectro_y_size)
spectro_z = np.zeros((spectro_y.shape[0], spectro_x.shape[0]))
spectro_z_min, spectro_z_max = SPECTROGRAM_MIN_VALUE, SPECTROGRAM_MAX_VALUE

ax_spectro.set_yscale('log')
ax_spectro.set_ylim(SPECTROGRAM_MIN_FREQUENCY, SPECTROGRAM_MAX_FREQUENCY)
ax_spectro.set_xlim(0, SPECTROGRAM_SECS)
plt.setp(ax_spectro, yticks=[10 ** i for i in range(0, math.ceil(math.log(RATE, 10)))])
spectro_heatmap = ax_spectro.pcolormesh(spectro_x, spectro_y, spectro_z, cmap='magma', vmin=spectro_z_min,
                                        vmax=spectro_z_max)
# fig.colorbar(spectro_heatmap, ax=ax_spectro)

"""***   THAYER MODEL   ***"""
emotion_detector = ThayerRandom()
ax_thayer.set_title('Thayer model')
ax_thayer.set_xlabel('Valance')
ax_thayer.set_ylabel('Arousal')
thayer_values = [(0.0, 0.0)] * THAYER_SCATTER_BUFFER_SIZE

thayer_scatter = ax_thayer.scatter([x for x, y in thayer_values], [y for x, y in thayer_values], c=thayer_scatter_colors, s=thayer_scatter_sizes)
# ax_thayer.grid(True, which='both')
ax_thayer.axhline(y=0, color='k')
ax_thayer.axvline(x=0, color='k')
ax_thayer.set_xlim(-1, 1)
ax_thayer.set_ylim(-1, 1)

"""***   EMOTIONS   ***"""
emotions: Dict[AimpathyEmotion, List[float]] = {emotion: list() for emotion in AimpathyEmotion}
emotions_graphs = dict()
emotions_graph_x_lim = 10
ax_emotion.set_title('Perceived emotion graph')
ax_emotion.set_xlabel('Time [sec]')
ax_emotion.set_ylabel('Emotion level [%]')
ax_emotion.set_ylim(0, 1)
ax_emotion.set_xlim(0, emotions_graph_x_lim)

for emotion in emotions.keys():
    emotion_line, = ax_emotion.plot([0.0], [0.0], '-', lw=3, label=emotion.name(), c=emotion.color())
    emotions_graphs[emotion] = emotion_line

ax_emotion.legend()

print("Display ready")
plt.show(block=False)

while True:
    # binary data
    data = stream.read(CHUNK, exception_on_overflow=False)
    data_int = struct.unpack(str(2 * CHUNK) + 'B', data)

    wave_data_np = np.array(data_int, dtype='b')[::2] + 128
    wave_data_np = smooth(wave_data_np, SMOOTH_RATE)
    wave_line.set_ydata(wave_data_np)

    snd_block = np.array(struct.unpack('%dh' % (len(data) / 2), data))
    f, t, Sxx = signal.spectrogram(snd_block, RATE, scaling="spectrum", nperseg=FFT_POINTS)
    spectro_z = np.roll(spectro_z, 2, axis=1)
    spectro_z[:, 0:2] = Sxx
    spectro_heatmap.set_array(spectro_z)

    new_thayer_x, new_thayer_y = emotion_detector.calculate(list(Sxx[1:, 0]))
    thayer_values.pop()
    thayer_values.insert(0, (new_thayer_x, new_thayer_y))
    thayer_scatter.set_offsets(np.c_[[x for x, y in thayer_values], [y for x, y in thayer_values]])

    new_emotion = thayer_coordinates_to_emotion(new_thayer_x, new_thayer_y)
    for emotion, emotion_value in new_emotion.items():
        emotions[emotion].append(emotion_value)
        emotions_graphs[emotion].set_xdata([i*SECS_PER_SPECTROGRAM_SEGMENT for i in range(len(emotions[emotion]))])
        emotions_graphs[emotion].set_ydata(emotions[emotion])
    if (len(list(emotions.values())[0]) * SECS_PER_SPECTROGRAM_SEGMENT) >= emotions_graph_x_lim:
        emotions_graph_x_lim += 10
        ax_emotion.set_xlim(0, emotions_graph_x_lim)

    try:
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(REFRESH_RATE)

    except:
        print('stream stopped')
        break

print('ended')
