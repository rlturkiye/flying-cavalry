from Camera import Camera
import airsim
import numpy as np
import os
import cv2
import datetime

class Depth(Camera):

    def __init__(self, depth_cam = "depth_vis"):
        """
        ????
        don't know the which camere we use. So we do this but we will delete when we learn.
        """
        depthCameraTypeMap = {
                        "depth_vis": airsim.ImageType.DepthVis,
                        "depth_perspective": airsim.ImageType.DepthPerspective,
                        "depth_planner": airsim.ImageType.DepthPlanner
                        }

        self.depth_cam = depthCameraTypeMap[depth_cam]

    def fetch_single_img(self, client, cam_type=0):
        """
        return  the numpy array
        cam type specifies the location of the camere in the drone.
        """

        self.client = client
        #Image request from AirSim
        response = self.client.simGetImages([airsim.ImageRequest(cam_type, self.depth_cam, pixels_as_float=True, compress=False)])[0]

        #get numpy array
        img1d = np.array(response.image_data_float, dtype=np.float32)
        # reshape array to 2 channel image array H X W x 1
        img_reshaped = img1d.reshape(response.height, response.width)
        img_depth = np.array(img_reshaped * 255, dtype=np.uint8)

        return img_depth


    def save_single_img(self, client,cam_type=0, file_name= "DEPTH "+str(datetime.datetime.now()), path="", format=".png"):
        """
        Saves the image to the specified location.
        cam_type: specifies the location of the camera in the drone.
        file_name:  gets the day and time the image was received by default,.
        path: specifies the location you want to save. It saves in the directory where it is located  by default.
        format: specifies the format of the picture (.pfm, .png etc.) takes ".png" by default
        """
        
        #Save an image on path with your format
        cv2.imwrite(os.path.join(path, file_name + format ), self.fetch_single_img(client=client))
        
        return 









        
        
