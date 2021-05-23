from DepthVision import DepthVision
from DepthPlanner import DepthPlanner
from DepthPerspective import DepthPerspective
from Segmentation import Segmentation
from RGB import RGB
import airsim
import cv2


client = airsim.MultirotorClient()

# rgb
sony = RGB(client=client, camera_name="0")

rgb_img = sony.fetch_single_img()
cv2.imshow("RGB_IMG", rgb_img)
cv2.waitKey(5555)
sony.save_single_img()
sony.camera_info()

# depth
panasonic = DepthVision(client=client, camera_name="DenemeCamera2")

depth_img = panasonic.fetch_single_img()
cv2.imshow("DEPTH_IMG", depth_img)
cv2.waitKey(5555)
panasonic.save_single_img()
panasonic.save_as_pfm(scale=99)
panasonic.camera_info()

# segmentation
canon = Segmentation(client=client, camera_name="DenemeCamera")

seg_img = canon.fetch_single_img()
cv2.imshow("SEG_IMG", seg_img)
cv2.waitKey(5555)
canon.save_single_img()
canon.camera_info()

airsim.CameraInfo()
