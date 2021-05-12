from CameraRl.Camera import Camera
import airsim
import numpy as np
import os
import cv2
import datetime
from record import Record

class DepthPerspective(Camera):
    """
    you get depth from camera using a projection ray that hits that pixel.
    """
    def __init__(self, client, camera_name):
        self.client = client
        self.camera_name = camera_name
    
    def fetch_single_img(self):
        """
        return  the numpy array
        cam type specifies the location of the camere in the drone.
        """
        #Image request from AirSim
        response = self.client.simGetImages([airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False)])[0]

        #get numpy array
        img1d = np.array(response.image_data_float, dtype=np.float32)
        # reshape array to 2 channel image array H X W x 1
        img_reshaped = img1d.reshape(response.height, response.width)
        img_depth = np.array(img_reshaped, dtype=np.float32)

        return img_depth


    def save_single_img(self, file_name= "DepthPerspective "+str(datetime.datetime.now()), path="./", format=".png"):
        """
        Saves the image to the specified location.
        Args:
            camera_name: specifies the location of the camera in the drone.
            file_name:  gets the day and time the image was received by default,.
            path: specifies the location you want to save. It saves in the directory where it is located  by default.
            format: specifies the format of the picture (.pfm, .png etc.) takes ".png" by default
        """
        
        #Save an image on path with your format
        Record.save_single_img(self,  file_name=file_name, path=path, format=format)    

        return 

    
    def save_as_pfm(self, scale=100, file_name="DepthPerspective PFM "+str(datetime.datetime.now()), path="./"):
        """
        Saves the image to the specified location with .pfm format.
        Args:
            cam_type: specifies the location of the camera in the drone.
            file_name:  gets the day and time the image was received by default,.
            path: specifies the location you want to save. It saves in the directory where it is located  by default.
            scale: ??????????????????
        """
        Record.save_as_pfm(self, scale=scale, file_name=file_name, path=path)

        return

    def camera_info(self,):
        """
        Get details about the camera
        Args:
            camera_name (str): Name of the camera, for backwards compatibility, ID numbers such as 0,1,etc. can also be used
        """
        info=self.client.simGetCameraInfo(self.camera_name)

        return print(info)