import airsim
import datetime
import cv2
import os
import numpy as np

class Record:

    def save_single_img(self, file_name="CamName "+str(datetime.datetime.now()), path="./", format=".png"):
        """
        Saves the image to the specified location.
        Args:
            cam_type: specifies the location of the camera in the drone.
            file_name:  gets the day and time the image was received by default,.
            path: specifies the location you want to save. It saves in the directory where it is located  by default.
            format: specifies the format of the picture (.jpg, .png etc.) takes ".png" by default
        """

        #Save an image on path with your format
        cv2.imwrite(os.path.join(path, file_name + format), self.fetch_single_img())

        return

    
    def save_as_pfm(self, scale=100, file_name="PFM "+str(datetime.datetime.now()), path="./"):
        """
        Saves the image to the specified location with .pfm format.
        Args:
            cam_type: specifies the location of the camera in the drone.
            file_name:  gets the day and time the image was received by default,.
            path: specifies the location you want to save. It saves in the directory where it is located  by default.
            scale: ??????????????????
        """
        airsim.write_pfm(os.path.join(path, file_name + ".pfm"), np.flipud(self.fetch_single_img()), scale=scale)

        return