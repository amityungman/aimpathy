from aimpathy_constants import *
from audio_utils import smooth
import pyaudio
import struct
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from scipy import signal


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
ax_thayer.set_title('Valance-Arousal graph (Thayer model)')
ax_thayer.set_xlabel('Valance')
ax_thayer.set_ylabel('Arousal')
# ax_spectro.set_ylim(0 - GRAPH_SHOULDERS, 255 + GRAPH_SHOULDERS)
# ax_spectro.set_xlim(0, 2 * CHUNK)
# plt.setp(ax_spectro, xticks=[0, CHUNK, 2 * CHUNK], yticks=[0, 128, 255])

"""***   EMOTIONS   ***"""
ax_emotion.set_title('Perceived emotion graph')
ax_emotion.set_xlabel('Time')
ax_emotion.set_ylabel('Emotion level')
ax_emotion.set_ylim(0 - GRAPH_SHOULDERS//5, 100 + GRAPH_SHOULDERS//5)
# ax_spectro.set_xlim(0, 2 * CHUNK)
# plt.setp(ax_spectro, xticks=[0, CHUNK, 2 * CHUNK], yticks=[0, 128, 255])

print("Display ready")
plt.show(block=False)

while True:
    # binary data
    data = stream.read(CHUNK, exception_on_overflow=False)
    data_int = struct.unpack(str(2 * CHUNK) + 'B', data)

    wave_data_np = np.array(data_int, dtype='b')[::2] + 128
    wave_data_np = smooth(wave_data_np, SMOOTH_RATE)
    wave_line.set_ydata(wave_data_np)

    count = len(data) / 2
    format = '%dh' % (count)
    snd_block = np.array(struct.unpack(format, data))
    f, t, Sxx = signal.spectrogram(snd_block, RATE, scaling="spectrum", nperseg=FFT_POINTS)
    spectro_z = np.roll(spectro_z, 2, axis=1)
    spectro_z[:, 0:2] = Sxx
    spectro_heatmap.set_array(spectro_z)

    # update figure canvas
    try:
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(REFRESH_RATE)

    except:
        print('stream stopped')
        break

print('ended')
