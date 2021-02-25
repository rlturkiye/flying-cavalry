from abc import ABC, abstractmethod
import airsim
import numpy as np
import os
import math

class Camera(ABC):

    @abstractmethod
    def show_single_img(self):
        pass

class RGB(Camera):


    def show_single_img(self):

        responses = client.simGetImages([
            airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])

        response = responses[0]

        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        img_rgb = img1d.reshape(response.height, response.width, 3)
        img_rgb = np.flipud(img_rgb)

        airsim.write_png(os.path.normpath( "RGB"+ ".tif"), img_rgb)
    
        return print("basarıli")


class Depth(Camera):

    def  show_single_img(self):
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPlanner, False, False)])

        response = responses[0]

        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        img_rgb = img1d.reshape(response.height, response.width, 3)
        img_rgb = np.flipud(img_rgb)

        airsim.write_png(os.path.normpath("DepthPlanner"+ ".tif"), img_rgb)
    
        return print("basarıli")


client = airsim.MultirotorClient()

a = RGB()
a.show_single_img()

b = Depth()
b.show_single_img()



