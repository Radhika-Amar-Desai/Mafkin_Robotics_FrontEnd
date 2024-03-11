import numpy as np
import cv2
vec1 = np.array ( [1,2,3,4,5,6,7] )
vec2 = np.array ( [2,3,4,5,6,7,8] )

image_path = r"VisualOdometry\feature_extraction\dataset_for_model\train\similar\image_pair1_blob0\image2\blob_at_2448_1924_image_2.jpg"
image = cv2.imread ( image_path )
gray_image = cv2.cvtColor ( image , cv2.COLOR_BGR2GRAY )
print ( gray_image.shape )