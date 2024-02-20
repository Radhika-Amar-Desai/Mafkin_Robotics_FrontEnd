import cv2

def match_orb_features(image1_path, image2_path):
    # Load images
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Initialize FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)  # or pass empty dictionary

    # Initialize FLANN matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Perform matching
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Ratio test to find good matches
    good_matches = []
    for match_pair in matches:
        if len(match_pair) < 2:
            continue
        m , n = match_pair
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Draw matches
    matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Resize the window
    cv2.namedWindow("Matched Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Matched Image", 800, 600)

    # Show the matched image
    cv2.imshow("Matched Image", matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Return matched keypoints
    matched_keypoints1 = [keypoints1[match.queryIdx].pt for match in good_matches]
    matched_keypoints2 = [keypoints2[match.trainIdx].pt for match in good_matches]

    return [ [keypoint, matched_keypoints2[index] ] \
            for index,keypoint in enumerate ( matched_keypoints1 )]

def mark_keypoint_with_region(image_path, keypoint, region_size):
    """
    Mark a keypoint with a yellow dot in the given image and mark the region around it with yellow color.

    Parameters:
    - image_path: Path to the input image.
    - keypoint_x: X-coordinate of the keypoint.
    - keypoint_y: Y-coordinate of the keypoint.
    - region_size: Size of the region around the keypoint.
    """
    # Load the image
    image = cv2.imread(image_path)

    # Define the color (BGR) for the dot and region (in this case, yellow)
    dot_color = (0, 255, 255)  # Yellow in BGR

    keypoint_x , keypoint_y = keypoint

    # Draw a dot at the keypoint coordinates
    cv2.circle(image, (keypoint_x, keypoint_y), 5, dot_color, -1)

    # Draw a rectangle to mark the region around the keypoint
    half_size = region_size // 2
    top_left = (keypoint_x - half_size, keypoint_y - half_size)
    bottom_right = (keypoint_x + half_size, keypoint_y + half_size)
    cv2.rectangle(image, top_left, bottom_right, dot_color, thickness=2)

    return image

image = mark_keypoint_with_region ( 
        r"VisualOdometry\feature_extraction\dataset\images\image_pair1\image1.jpg",
        (100,150), 50 )

# Create a named window and resize it according to the image size
cv2.namedWindow("window_name", cv2.WINDOW_NORMAL)
cv2.resizeWindow("window_name", 800, 600)
# Display the image with the marked keypoint and region
cv2.imshow("window_name", image)
cv2.waitKey(0)
cv2.destroyAllWindows()