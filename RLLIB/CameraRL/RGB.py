from CameraRL.Camera import Camera
import airsim
import numpy as np
import os
import cv2
import datetime
from CameraRL.record import Record
from PIL import Image

class RGB(Camera):


    def __init__(self, client, camera_name="0"):

        self.client = client
        self.camera_name = camera_name


    def fetch_single_img(self):
        """
        Returns the numpy array.
        camera_name: specifies the location of the camera in the drone. If you define camera from settings.json, this is your camera name.
        """
        #Image request from AirSim
        response = self.client.simGetImages([airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene, pixels_as_float=False, compress=False)])[0]
        # get numpy array
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        # reshape array to 3 channel image array H X W X 3
        img_rgb = img1d.reshape(response.height, response.width, 3)

        img_rgb_resized = Image.fromarray(img_rgb).resize((84, 84)).convert("L")
        img_rgb = np.array(img_rgb_resized).reshape(84, 84, 1)

        return img_rgb

    
    def save_single_img(self, file_name="RGB "+str(datetime.datetime.now()), path="./", format=".png"):
        """
        Saves the image to the specified location.
        Args:
            camera_name: specifies the location of the camera in the drone.
            file_name:  gets the day and time the image was received by default,.
            path: specifies the location you want to save. It saves in the directory where it is located  by default.
            format: specifies the format of the picture (.jpg, .png etc.) takes ".png" by default
        """
        Record.save_single_img(self, file_name=file_name, path=path, format=format)
        #Save an image on path with your format
        return
    
    def camera_info(self):
        """
        Get details about the camera
        Args:
            camera_name specifies the location of the camera in the drone. If you define camera from settings.json, this is your camera name.
        """
        info= self.client.simGetCameraInfo(self.camera_name)

        return print(info)
