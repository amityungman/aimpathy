from typing import Tuple, List
from models.thayer_detector import ThayerModel
from aimpathy_constants import *
from statistics import mean


class ThayerNaive(ThayerModel):
    AROUSAL_LOG_ROOT: float
    _pitch_range: Tuple[float, float, float]
    _volume_range: Tuple[float, float]
    _spectrogram = np.linspace(0, RATE // 2, FFT_POINTS_UNSIGNED + 1)
    _last_values: List[Tuple[float, float]] = list()
    _smooth_factor: int

    def __init__(self, min_pitch: float = 87.30706, mid_pitch: float = 220, max_pitch: float = 1760.0,
                 min_volume: float = 100, max_volume: float = 10**6, volume_log_root: float = 2, smooth_factor: int = 10):
        self._pitch_range = min_pitch, mid_pitch, max_pitch
        self._volume_range = min_volume, max_volume
        self.AROUSAL_LOG_ROOT = volume_log_root
        self._smooth_factor = smooth_factor

    def get_valance(self, pitch: float) -> float:
        pmin, pmid, pmax = self._pitch_range
        if pitch <= pmin:
            return -1.0
        if pitch >= pmax:
            return 1.0
        if pmin < pitch <= pmid:
            return ((pitch - pmin) / (pmid - pmin)) - 1
        return (pitch - pmid) / (pmax - pmid)

    def get_arousal(self, volume: float) -> float:
        arousal_raw_value = 2 * ((volume - self._volume_range[0]) ** self.AROUSAL_LOG_ROOT) / \
                            ((self._volume_range[1] - self._volume_range[0]) ** self.AROUSAL_LOG_ROOT) - 1
        return min(max(arousal_raw_value, -1.0), 1.0)

    def calculate(self, spectrogram: List[float]) -> Tuple[float, float]:
        max_volume = max(spectrogram)
        if max_volume < self._volume_range[0]:
            return 0.0, 0.0
        pitch_ind = spectrogram.index(max_volume)
        pitch = self._spectrogram[pitch_ind]
        valance = self.get_valance(pitch)

        arousal = self.get_arousal(max_volume)
        print(f"pitch: {pitch} | valance: {valance} | volume: {max(spectrogram)} | arousal: {arousal}")

        if len(self._last_values) >= self._smooth_factor:
            self._last_values = self._last_values[1:]
        self._last_values.append((valance, arousal))
        return mean([val for val, aro in self._last_values]), mean([aro for val, aro in self._last_values])
