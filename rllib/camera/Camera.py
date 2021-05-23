from abc import ABC, abstractmethod
import datetime
import airsim


class Camera(ABC):

    @abstractmethod
    # Returns the numpy array
    def fetch_single_img(self):
        pass

    @abstractmethod
    # Saves the image to the specified location.
    def save_single_img(self):
        pass

    @abstractmethod
    # Get camera info
    def camera_info(self):
        pass
