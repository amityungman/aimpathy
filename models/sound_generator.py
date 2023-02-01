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


note_A = AudioData()
note_A.add_sound([69], 0, 1, 100)
note_A.save_to_wav(os.sep.join(["..", "data", "test_audio", "A.wav"]))
