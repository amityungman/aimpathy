from typing import Tuple, List
from models.thayer_detector import ThayerModel
from aimpathy_constants import *
from statistics import mean


class ThayerNaive(ThayerModel):
    AROUSAL_LOG_ROOT: float
    _pitch_range: Tuple[float, float, float]
    _volume_range: Tuple[float, float]
    _spectrogram = np.linspace(0, RATE // 2, FFT_POINTS_UNSIGNED + 1)
    _last_value: Tuple[float, float] = (0.0, 0.0)

    def __init__(self, min_pitch: float = 87.30706, mid_pitch: float = 261.6256, max_pitch: float = 1760.0,
                 min_volume: float = 0, max_volume: float = 10**6, volume_log_root: float = 3):
        self._pitch_range = min_pitch, mid_pitch, max_pitch
        self._volume_range = min_volume, max_volume
        self.AROUSAL_LOG_ROOT = volume_log_root

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
        pitch_ind = spectrogram.index(max(spectrogram))
        pitch = self._spectrogram[pitch_ind]
        valance = self.get_valance(pitch)

        arousal = self.get_arousal(max(spectrogram))
        print(f"pitch: {pitch} | valance: {valance} | volume: {max(spectrogram)} | arousal: {arousal}")

        self._last_value = mean([arousal, self._last_value[0]]), mean([valance, self._last_value[1]])
        return self._last_value
