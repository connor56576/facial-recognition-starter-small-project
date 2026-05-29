from abc import ABC, abstractmethod
import numpy as np

"""
 Base class for effect groups. Each group handles its own gesture parsing and effect logic, and provides a list of effects it can apply. 
 This allows the main loop to be generic and just call update on the current group without needing to know any details about how they work.
 I'm mostly including this for future expansion with other effect groups that i intend to add. And plus this is much cleaner.
 I'm trying to write more modular code in general and this is a good start.

 """

class BaseEffectGroup(ABC):

    @property
    @abstractmethod
    def effect_list(self) -> list:
        """list of effect name strings this group provides."""
        ...

    @abstractmethod
    def process_landmarks(self, results, frame_shape: tuple) -> dict:
        """Convert mediapipe results into gesture data for this group."""
        ...

    @abstractmethod
    def update(self, gesture_data: dict, now: float) -> bool:
        """Handle gesture logic. Return True if effect should switch ."""
        ...

    @abstractmethod
    def process_frame(self, frame: np.ndarray, effect_name: str) -> np.ndarray:
        """Apply effect to frame."""
        ...