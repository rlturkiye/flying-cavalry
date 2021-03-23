from RGB import RGB
from Segmentation import Segmentation

class DenemeCamera():

    def __init__(self, scene_type):
        self.scene_type = scene_type

    def fetch_single_img(self, client, camera_name="0"):

        if self.scene_type == "RGB":
            return RGB.fetch_single_img(self, client=client , camera_name=camera_name)
        elif self.scene_type == "Segmentation":
            return Segmentation.fetch_single_img(self, client=client , camera_name=camera_name)
    
