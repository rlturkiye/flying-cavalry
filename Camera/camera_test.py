from Depth import Depth
from RGB import RGB
import airsim
import cv2

client = airsim.MultirotorClient()


# RGB 
#declaret RGB camera
sony = RGB()
#get sample image for RGB
rgb_sample_img = sony.fetch_single_img(client=client)
#show sample image for RGB with cv2
cv2.imshow("RGB",rgb_sample_img)
#waitkey for the image to wait on the screen for a while
cv2.waitKey(5555)
#for save the image
sony.save_single_img(client = client, cam_type="2", file_name="RGB_Example", path=" ", format=".png")
#for get camera info
sony.camera_info(client=client)

# Depth
#declaret Depth camera
panasonic = Depth()
#get sample image for Depth
depth_sample_img = panasonic.fetch_single_img(client)
#show sample image for Depth with cv2
cv2.imshow("Depth",depth_sample_img)
#waitkey for the image to wait on the screen for a while
cv2.waitKey(5555)
#for save the image
panasonic.save_single_img(client = client, cam_type="4", file_name="Depth_Example", path=" ", format=".pfm")
#for get camera info
panasonic.camera_info(client=client,cam_type="4")

#Displaying the image with the .pfm extension
depth_pfm = cv2.imread("/home/caner/Pictures/test/Depth_Example.pfm")
cv2.imshow("DepthPfm",depth_sample_img)
cv2.waitKey(5555)