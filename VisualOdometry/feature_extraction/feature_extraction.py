import cv2
import numpy as np
import utils as utils

def find_corresponding_orb_features(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect ORB features and compute descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Create a BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches based on their distances
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw matches
    # Draw matches with more visible lines
# Draw matches with more visible lines
    
    # matched_image = cv2.drawMatches(image1, keypoints1, 
    #                             image2, keypoints2, matches[:10], None, 
    #                             matchColor=(0, 255, 0),  # Green lines
    #                             singlePointColor=(0, 0, 255),  # Red keypoints
    #                             matchesMask=None, 
    #                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the matched image
    # cv2.namedWindow("Matched Features", cv2.WINDOW_NORMAL)
    # cv2.imshow("Matched Features", matched_image)
    # cv2.resizeWindow("Matched Features", 800, 600)  # Adjust the size here
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Extract corresponding keypoints
    corresponding_keypoints = [(keypoints1[match.queryIdx], 
                                keypoints2[match.trainIdx]) for match in matches] 

    return corresponding_keypoints

def get_blob(image, key_point, blob_size):
    """
    Extracts a patch (blob) from the image centered at the key_point
    with the specified blob_size (height, width).

    Args:
    - image: The input image.
    - key_point: The key point (x, y) around which the blob is extracted.
    - blob_size: The size of the blob in pixels (height, width).

    Returns:
    - blob_patch: The extracted patch (blob) from the image.
    """

    # Extracting key point coordinates
    kp_x, kp_y = int(key_point[0]), int(key_point[1])

    # Extracting blob size
    blob_height, blob_width = blob_size

    # Calculating coordinates for the top-left corner of the blob
    top_left_x = max(0, kp_x - blob_width // 2)
    top_left_y = max(0, kp_y - blob_height // 2)

    # Calculating coordinates for the bottom-right corner of the blob
    bottom_right_x = min(image.shape[1], kp_x + blob_width // 2)
    bottom_right_y = min(image.shape[0], kp_y + blob_height // 2)

    # Extracting the blob patch from the image
    blob_patch = image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    return blob_patch

