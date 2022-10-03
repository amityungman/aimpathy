from typing import Tuple, Dict
from enum import Enum


class AimpathyEmotion(Enum):
    _title: str
    _thayer_box: Tuple[Tuple[float, float], Tuple[float, float]]  # (x_min, x_max), (y_min, y_max)
    _thayer_peak: Tuple[float, float]  # (x_max, y_max)
    _color: str

    def __init__(self, title: str, thayer_box: Tuple[Tuple[float, float], Tuple[float, float]],
                 color: str = None, thayer_peak: Tuple[float, float] = None):
        self._title = title
        self._color = color
        self._thayer_box = ((0.0, 0.0), (0.0, 0.0))
        if thayer_box and (-1 <= thayer_box[0][0] <= 1) and (-1 <= thayer_box[0][1] <= 1) and \
                (-1 <= thayer_box[1][0] <= 1) and (-1 <= thayer_box[1][1] <= 1):
            self._thayer_box = thayer_box
        else:
            if not thayer_box:
                ValueError(f"The thayer box was empty")
            raise ValueError(f"The thayer box must have 4 float values in the range [-1, 1]. Got: {thayer_box}")

        if not thayer_peak:
            self._thayer_peak = (thayer_box[0][1] if abs(thayer_box[0][1]) > abs(thayer_box[0][0]) else thayer_box[0][0],
                                 thayer_box[1][1] if abs(thayer_box[1][1]) > abs(thayer_box[1][0]) else thayer_box[1][0])
        else:
            self._thayer_peak = thayer_peak

    def name(self) -> str:
        return self._title

    def x_min(self) -> float:
        return self._thayer_box[0][0]

    def x_max(self) -> float:
        return self._thayer_box[0][1]

    def y_min(self) -> float:
        return self._thayer_box[1][0]

    def y_max(self) -> float:
        return self._thayer_box[1][1]

    def peak(self) -> Tuple[float, float]:
        return self._thayer_peak

    def in_range(self, x: float, y: float) -> bool:
        return self.x_min() <= x <= self.x_max() and self.y_min() <= y <= self.y_max()

    def color(self):
        return self._color

    HAPPY = ("Happy", ((0.0, 1.0), (0.0, 1.0)), "#ffcc00")
    ANGRY = ("Angry", ((-1.0, 0.0), (0.0, 1.0)), "#ff3333")
    SAD = ("Sad", ((-1.0, 0.0), (-1.0, 0.0)), "#0066ff")
    RELAXED = ("Relaxed", ((0.0, 1.0), (-1.0, 0.0)), "#009933")


MAX_DIST = ((2 ** 2) + (2 ** 2)) ** 0.5


def thayer_coordinates_to_emotion(thayer_x: float, thayer_y: float, allow_out_of_box: bool = True) -> Dict[AimpathyEmotion, float]:
    """
    Calculate the emotions' values based on the thayer value
    :param thayer_x: The x value (Valance)
    :param thayer_y: The y value (Arousal)
    :return: A mapping between emotion and its level in the range [0, 1]
    """
    results: Dict[AimpathyEmotion, float] = dict()
    for emotion in AimpathyEmotion:
        ratio = 0.0
        if not allow_out_of_box:
            if emotion.in_range(thayer_x, thayer_y):
                min_x, min_y = min(abs(emotion.x_min()), abs(emotion.x_max())), min(abs(emotion.y_min()), abs(emotion.y_max()))
                max_x, max_y = max(abs(emotion.x_min()), abs(emotion.x_max())), max(abs(emotion.y_min()), abs(emotion.y_max()))

                distance = ((thayer_x - min_x) ** 2 + (thayer_y - min_y) ** 2) ** 0.5
                max_distance = ((max_x - min_x) ** 2 + (max_y - min_y) ** 2) ** 0.5
                ratio = distance / max_distance
        else:
            max_x, max_y = emotion.peak()
            distance = ((thayer_x - max_x) ** 2 + (thayer_y - max_y) ** 2) ** 0.5
            ratio = (1 - (distance / MAX_DIST)) ** 4
        results[emotion] = ratio
    return results
