from Depth import Depth
from RGB import RGB
import airsim
import cv2

client = airsim.MultirotorClient()


# RGB 
sony = RGB()
rgb_sample_img = sony.fetch_single_img(client=client)
cv2.imshow("kek",rgb_sample_img)
cv2.waitKey(5555)
sony.save_single_img(client = client, cam_type="2", file_name="RGB_Example", path="/home/caner/Pictures/test", format=".png")

# Depth
panasonic = Depth()
depth_sample_img = panasonic.fetch_single_img(client)
cv2.imshow("Depth",depth_sample_img)
cv2.waitKey(5555)
panasonic.save_single_img(client = client, cam_type="4", file_name="Depth_Example", path="/home/caner/Pictures/test", format=".pfm")


depth_pfm = cv2.imread("/home/caner/Pictures/test/Depth_Example.pfm")
cv2.imshow("inş",depth_sample_img)
cv2.waitKey(5555)