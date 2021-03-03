from Camera import Camera
import airsim
import numpy as np
import os
import cv2
import datetime


class RGB(Camera):

    def fetch_single_img(self, client, cam_type="0"):
        """
        Returns the numpy array.
        cam_type specifies the location of the camera in the drone.
        """
        self.client = client
        #Image request from AirSim
        response = client.simGetImages([airsim.ImageRequest(cam_type, airsim.ImageType.Scene, pixels_as_float=False, compress=False)])[0]
        # get numpy array
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 
        # reshape array to 3 channel image array H X W X 3
        img_rgb = img1d.reshape(response.height, response.width, 3)

        return img_rgb

    
    def save_single_img(self, client, cam_type=0, file_name="RGB "+str(datetime.datetime.now()), path="./", format=".png"):
        """
        Saves the image to the specified location.
        cam_type: specifies the location of the camera in the drone.
        file_name:  gets the day and time the image was received by default,.
        path: specifies the location you want to save. It saves in the directory where it is located  by default.
        format: specifies the format of the picture (.jpg, .png etc.) takes ".png" by default
        """

        #Save an image on path with your format
        cv2.imwrite(os.path.join(path, file_name + format), self.fetch_single_img(client))

        return
    
