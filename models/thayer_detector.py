from typing import Tuple
import random


class ThayerModel(object):
    def calculate(self) -> Tuple[float, float]:
        pass


class ThayerRandom(ThayerModel):
    _max_step_size = 0.1
    _last_value_x = (random.random() * 2 - 1)
    _last_value_y = (random.random() * 2 - 1)

    def calculate(self) -> Tuple[float, float]:
        self._last_value_x = self._last_value_x + self._max_step_size * (random.random() * 2 - 1)
        self._last_value_x = 1 if self._last_value_x > 1 else self._last_value_x
        self._last_value_x = -1 if self._last_value_x < -1 else self._last_value_x
        self._last_value_y = self._last_value_y + self._max_step_size * (random.random() * 2 - 1)
        self._last_value_y = 1 if self._last_value_y > 1 else self._last_value_y
        self._last_value_y = -1 if self._last_value_y < -1 else self._last_value_y
        return self._last_value_x, self._last_value_y
