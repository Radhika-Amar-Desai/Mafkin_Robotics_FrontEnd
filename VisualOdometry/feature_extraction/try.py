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

# Example usage:
if __name__ == "__main__":
    # Provide the paths to the images
    image1_path = \
        r"VisualOdometry\feature_extraction\dataset\images\image_pair2\image1.jpg"
    image2_path = \
        r"VisualOdometry\feature_extraction\dataset\images\image_pair2\image2.jpg"
    
    # Match ORB features
    num_matches = match_orb_features(image1_path, image2_path)

    # Print the number of matches
    print("Number of matches:", num_matches)
