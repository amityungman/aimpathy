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

fig, axes = plt.subplot_mosaic("AAB;CDD", figsize=(20, 7))
ax_wave, ax_specto, ax_thayer, ax_emotion = axes.values()

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
ax_specto.set_title('Spectrogram')
ax_specto.set_xlabel('Time [sec]')
ax_specto.set_ylabel('Frequency [Hz]')
# specto_y, specto_x = np.meshgrid(np.linspace(0, RATE, RATE//10),
#                                  np.linspace(0, SPECTROGRAM_SECS * 60, int((SPECTROGRAM_SECS * 60) / REFRESH_RATE)))
# specto_z = np.zeros(specto_x.shape)
# specto_z = specto_z[:-1, :-1]
# specto_z_min, specto_z_max = 0, 256
#
# ax_specto.set_yscale('log')
# ax_specto.set_ylim(1, RATE)
# ax_specto.set_xlim(0, SPECTROGRAM_SECS * 60)
# plt.setp(ax_specto, xticks=np.linspace(0, SPECTROGRAM_SECS * 60, (SPECTROGRAM_SECS * 60 // 15) + 1),
#          yticks=[10**i for i in range(0, math.ceil(math.log(RATE, 10)))])
# specto_heatmap = ax_specto.pcolormesh(specto_x, specto_y, specto_z, cmap='viridis', vmin=specto_z_min, vmax=specto_z_max)
# fig.colorbar(specto_heatmap, ax=ax_specto)

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
    data = stream.read(CHUNK, exception_on_overflow=False)
    data_int = struct.unpack(str(2 * CHUNK) + 'B', data)

    wave_data_np = np.array(data_int, dtype='b')[::2] + 128
    wave_data_np = smooth(wave_data_np, SMOOTH_RATE)
    wave_line.set_ydata(wave_data_np)

    count = len(data) / 2
    format = '%dh' % (count)
    snd_block = np.array(struct.unpack(format, data))
    f, t, Sxx = signal.spectrogram(snd_block, RATE, scaling="spectrum", nperseg=CHUNK//2)
    specto_heatmap = ax_specto.pcolormesh(t, f, Sxx, cmap='viridis')
    plt.setp(ax_specto, yticks=[10**i for i in range(0, math.ceil(math.log(RATE, 10)))])
    ax_specto.set_yscale('log')
    ax_specto.set_ylim(20, RATE//3)
    # specto_data = np.fft.fft(data_int)
    # specto_z = np.roll(specto_z, 1, axis=0)
    # specto_z[0] = specto_data
    # specto_heatmap.set_zdata(specto_z)

    # update figure canvas
    try:
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(REFRESH_RATE)

    except:
        print('stream stopped')
        break

print('ended')
