import os
from typing import List
import midiutil
from collections import defaultdict
import mido
from pydub import AudioSegment
from pydub.generators import Sine


class AudioData(object):
    channels = None
    tempo = None  # In BPM
    midi: midiutil.MIDIFile = None

    def __init__(self, tempo: int = 120, tracks: int = 1, channels: int = 1):
        self.channels = channels
        self.tempo = tempo
        self.midi = midiutil.MIDIFile(tracks)
        self.midi.addTempo(0, 0, tempo)

    def add_sound(self, notes: List[int], time: int, duration: int, volume: int, track: int = 0, channel: int = 0):
        """
        :param notes: The notes to add, 0-127
        https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies
        :param time: start time in beats
        :param duration: duration in beats
        :param volume: 0-127, as per the MIDI standard
        :param track: The track to add to
        :param channel: The channel to add to
        """
        self.midi.addTempo(track, time, self.tempo)
        for note in notes:
            self.midi.addNote(track, channel, note, time, duration, volume)

    def save_to_wav(self, output_file_name: str):
        temp_file_name = f"{output_file_name}.mid"
        with open(temp_file_name, "wb") as midi_output_file:
            self.midi.writeFile(midi_output_file)

        mid = mido.MidiFile(temp_file_name)
        output = AudioSegment.silent(mid.length * 1000.0)
        for track in mid.tracks:
            current_pos = 0.0
            current_notes = defaultdict(dict)

            for msg in track:
                current_pos += self.ticks_to_ms(mid, msg.time)

                if msg.type == 'note_on':
                    current_notes[msg.channel][msg.note] = (current_pos, msg)

                if msg.type == 'note_off':
                    start_pos, start_msg = current_notes[msg.channel].pop(msg.note)

                    duration = current_pos - start_pos

                    signal_generator = Sine(self.note_to_freq(msg.note))
                    rendered = signal_generator.to_audio_segment(duration=duration - 50, volume=-20).fade_out(100).fade_in(30)

                    output = output.overlay(rendered, start_pos)
        output.export(output_file_name, format="wav")
        os.remove(temp_file_name)

    def ticks_to_ms(self, mid, ticks):
        tick_ms = (60000.0 / self.tempo) / mid.ticks_per_beat
        return ticks * tick_ms

    @classmethod
    def note_to_freq(cls, note, concert_A=440.0):
        return (2.0 ** ((note - 69) / 12.0)) * concert_A


###    SCRIPT

# chromatic_scale_up = AudioData()
# chromatic_scale_down = AudioData()
# for i in range(50):
#     chromatic_scale_up.add_sound([50+i], i, 1, 100)
#     chromatic_scale_down.add_sound([100-i], i, 1, 100)
# chromatic_scale_up.save_to_wav(os.sep.join(["..", "data", "test_audio", "chromatic_scale_up.wav"]))
# chromatic_scale_down.save_to_wav(os.sep.join(["..", "data", "test_audio", "chromatic_scale_down.wav"]))
#
#
# chromatic_scale_up = AudioData()
# chromatic_scale_down = AudioData()
# for i in range(100):
#     chromatic_scale_up.add_sound([50+i//2], i, 1, 100)
#     chromatic_scale_down.add_sound([100-i//2], i, 1, 100)
# chromatic_scale_up.save_to_wav(os.sep.join(["..", "data", "test_audio", "chromatic_double_scale_up.wav"]))
# chromatic_scale_down.save_to_wav(os.sep.join(["..", "data", "test_audio", "chromatic_double_scale_down.wav"]))
#
#
# chromatic_scale_up = AudioData()
# chromatic_scale_down = AudioData()
# for i in range(4*50):
#     chromatic_scale_up.add_sound([50+i//4], i, 1, 100)
#     chromatic_scale_down.add_sound([100-i//4], i, 1, 100)
# chromatic_scale_up.save_to_wav(os.sep.join(["..", "data", "test_audio", "chromatic_4_scale_up.wav"]))
# chromatic_scale_down.save_to_wav(os.sep.join(["..", "data", "test_audio", "chromatic_4_scale_down.wav"]))
#


# one_tone_A = AudioData()
# one_tone_A.add_sound([69], 0, 50, 100)
# one_tone_A.save_to_wav(os.sep.join(["..", "data", "test_audio", "one_tone_A.wav"]))
#
# one_tone_C = AudioData()
# one_tone_C.add_sound([60], 0, 50, 100)
# one_tone_C.save_to_wav(os.sep.join(["..", "data", "test_audio", "one_tone_C.wav"]))
#
#
# chromatic_scale_up = AudioData()
# chromatic_scale_down = AudioData()
# for i in range(8*50):
#     chromatic_scale_up.add_sound([40+i//8], i, 1, 100)
#     chromatic_scale_down.add_sound([90-i//8], i, 1, 100)
# chromatic_scale_up.save_to_wav(os.sep.join(["..", "data", "test_audio", "chromatic_8_scale_up.wav"]))
# chromatic_scale_down.save_to_wav(os.sep.join(["..", "data", "test_audio", "chromatic_8_scale_down.wav"]))


# one_tone_A_120_bpm = AudioData()
# one_tone_C_120_bpm = AudioData()
# for i in range(2*25):
#     one_tone_A_120_bpm.add_sound([69], i, 1, 100 if (i % 2 == 0) else 0)
#     one_tone_C_120_bpm.add_sound([60], i, 1, 100 if (i % 2 == 0) else 0)
# one_tone_A_120_bpm.save_to_wav(os.sep.join(["..", "data", "test_audio", "one_tone_A_120_bpm.wav"]))
# one_tone_C_120_bpm.save_to_wav(os.sep.join(["..", "data", "test_audio", "one_tone_C_120_bpm.wav"]))


Harmony_A_C_one_tone = AudioData()
Harmony_A_C_120_bpm = AudioData()
Harmony_A_Cs_one_tone = AudioData()
Harmony_A_Cs_120_bpm = AudioData()
Harmony_A_C_one_tone.add_sound([69, 72], 0, 50, 100)
Harmony_A_Cs_one_tone.add_sound([69, 73], 0, 50, 100)
for i in range(2*25):
    Harmony_A_C_120_bpm.add_sound([69, 72], i, 1, 100 if (i % 2 == 0) else 0)
    Harmony_A_Cs_120_bpm.add_sound([69, 73], i, 1, 100 if (i % 2 == 0) else 0)
Harmony_A_C_one_tone.save_to_wav(os.sep.join(["..", "data", "test_audio", "Harmony_A_C_one_tone.wav"]))
Harmony_A_C_120_bpm.save_to_wav(os.sep.join(["..", "data", "test_audio", "Harmony_A_C_120_bpm.wav"]))
Harmony_A_Cs_one_tone.save_to_wav(os.sep.join(["..", "data", "test_audio", "Harmony_A_Cs_one_tone.wav"]))
Harmony_A_Cs_120_bpm.save_to_wav(os.sep.join(["..", "data", "test_audio", "Harmony_A_Cs_120_bpm.wav"]))




