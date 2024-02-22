import cv2
import numpy as np
import utils as utils
import os
import random
import shutil
import torch

def find_corresponding_orb_features(image1, image2):
    # Load images
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

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
    # matched_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # # Resize the window
    # cv2.namedWindow("Matched Image", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Matched Image", 800, 600)

    # # Show the matched image
    # cv2.imshow("Matched Image", matched_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # Return matched keypoints
    matched_keypoints1 = [keypoints1[match.queryIdx].pt for match in good_matches]
    matched_keypoints2 = [keypoints2[match.trainIdx].pt for match in good_matches]

    return [ [keypoint, matched_keypoints2[index] ] \
            for index,keypoint in enumerate ( matched_keypoints1 )]

def get_blob(image : list, key_point : tuple , blob_size : tuple):
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

def mark_keypoint_and_region(image_path, keypoint, region_size):
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

def generate_dataset_blobs_from_image ( image_pair : list,
                                    folder_path_to_save_blobs : str):
    
    image1 , image2 = image_pair

    def get_file_name_for_blob ( key_point : tuple, image_index : int, extension = "jpg" ):
        return "blob_at_" + str ( int(key_point[0]) ) + "_" +\
                str ( int(key_point[1]) ) + "_image_" + \
                str ( image_index ) + "." + extension

    def save_blob_and_image ( blob : list , image : list, key_point : tuple, \
                   image_index : int, folder_name : str):
        
        image_folder = os.path.join ( folder_name, "image" + str(image_index) )
        
        if not os.path.exists ( image_folder ): os.makedirs ( image_folder )

        blob_file_path = os.path.join ( image_folder , 
                                get_file_name_for_blob ( key_point, image_index ) )
        cv2.imwrite ( blob_file_path , blob )
        image_file_path = os.path.join ( image_folder ,
                                "image" + str(image_index) + ".jpg" )
        cv2.imwrite ( image_file_path , image )
        
    keypoints = find_corresponding_orb_features( image1 , image2 ) 

    for index , key_pt in enumerate ( keypoints ):
        blob_folder = os.path.join ( folder_path_to_save_blobs, \
                                    "blob" + str(index))
        
        if not os.path.exists ( blob_folder ): os.makedirs ( blob_folder )

        img0 = get_blob ( image1 , ( key_pt [0][0] , key_pt [0][1] ) , ( 50 , 50 ) )
        save_blob_and_image ( img0 , image1 , key_pt[0], 1, blob_folder )
        
        img1 = get_blob ( image2 , ( key_pt [1][0] , key_pt[1][1] ) , ( 50 , 50 ) )
        save_blob_and_image ( img1 , image2 , key_pt[1], 2, blob_folder )
    
    print ( "Done :)" )

def generate_dataset_blobs_from_image_folders ( 
        folder_for_image_pairs,
        folder_to_save_blobs
):
    for index, image_pair_folder in enumerate ( 
                    os.listdir ( folder_for_image_pairs ) ):
        
        image_pair_folder_path = os.path.join ( folder_for_image_pairs ,
                                               image_pair_folder )

        image_pair = []
        for idx, image_file in enumerate ( os.listdir ( 
                                        image_pair_folder_path ) ): 
            
            image_file_path = os.path.join ( image_pair_folder_path , 
                                            image_file )
            image_pair.append ( cv2.imread ( image_file_path ) )

        blobs_of_image_pair_folder_path = \
            os.path.join ( folder_to_save_blobs , 
                        "image_pair" + str ( index + 1 ))

        if not os.path.exists ( blobs_of_image_pair_folder_path ):
            os.makedirs ( blobs_of_image_pair_folder_path )

        generate_dataset_blobs_from_image ( image_pair , 
                                        blobs_of_image_pair_folder_path )

    print ( "Done :)" )

def generate_similar_folder ( folder_for_blobs : str, similar_folder_path : str ):
    
    if not os.path.exists ( similar_folder_path ): os.makedirs ( similar_folder_path )
    
    for image_pair_folder in os.listdir ( folder_for_blobs ):
        
        image_pair_folder_path = os.path.join ( folder_for_blobs , 
                                               image_pair_folder )
        
        for blob_folder in os.listdir ( image_pair_folder_path ):
            
            new_folder_name = image_pair_folder + "_" + blob_folder
            
            blob_folder_path = os.path.join ( image_pair_folder_path, blob_folder )
            new_folder_path = os.path.join ( similar_folder_path , new_folder_name )

            shutil.copytree ( blob_folder_path , new_folder_path )
        
    print ( "Done :)" )

def swap_image_files(file_path_1, file_path_2):
    # Open both image files in binary read mode and read their contents
    with open(file_path_1, 'rb') as file1, open(file_path_2, 'rb') as file2:
        file1_content = file1.read()
        file2_content = file2.read()
    
    # Open both image files in binary write mode and swap their contents
    with open(file_path_1, 'wb') as file1, open(file_path_2, 'wb') as file2:
        file1.write(file2_content)
        file2.write(file1_content)

def generate_dissimilar_folder ( dissimilar_folder_path : str ):
    
    def swapping_blobs_in_image_pair ( image_pair_path : str):
        
        blob_folders_name =  os.listdir ( image_pair_path )
        
        for index, blob_folder in enumerate ( blob_folders_name ):

            if index == 0:
                    next_blob_folder = blob_folders_name [ 0 ]
                    next_blob_folder_path = os.path.join ( image_pair_path, 
                                                        next_blob_folder )
                        
                    next_blob_image2_folder_path = os.path.join ( next_blob_folder_path,
                                                                "image2" )
                    
                    next_blob_image2_file = [ file for file in \
                                            os.listdir ( 
                                                next_blob_image2_folder_path )\
                                            if "blob" in file][0]
                    
                    next_blob_image2_file_path = os.path.join ( next_blob_image2_folder_path,
                                                            next_blob_image2_file )
                    curr_blob_folder_path = os.path.join ( image_pair_path, 
                                                        blob_folders_name [ -1 ] )
                    
                    curr_blob_image2_folder_path = os.path.join ( 
                                                        curr_blob_folder_path,
                                                        "image2" )
                    curr_blob_image2_file = [ file for file in \
                                                os.listdir ( 
                                                    curr_blob_image2_folder_path )\
                                                if "blob" in file ][0]

                    curr_blob_image2_file_path = os.path.join ( curr_blob_image2_folder_path,
                                                            curr_blob_image2_file )
                    
                    swap_image_files ( file_path_1 = next_blob_image2_file_path,
                                        file_path_2 = curr_blob_image2_file_path )

            elif 0 < index < len ( blob_folders_name ) - 1:
                next_blob_folder = blob_folders_name [ index + 1 ]
                next_blob_folder_path = os.path.join ( image_pair_path, 
                                                    next_blob_folder )
                    
                next_blob_image2_folder_path = os.path.join ( next_blob_folder_path,
                                                            "image2" )
                
                next_blob_image2_file = [ file for file in \
                                        os.listdir ( 
                                            next_blob_image2_folder_path )\
                                        if "blob" in file][0]
                
                next_blob_image2_file_path = os.path.join ( next_blob_image2_folder_path,
                                                        next_blob_image2_file )
                curr_blob_folder_path = os.path.join ( image_pair_path, 
                                                        blob_folder )
                
                curr_blob_image2_folder_path = os.path.join ( 
                                                    curr_blob_folder_path,
                                                    "image2" )
                curr_blob_image2_file = [ file for file in \
                                            os.listdir ( 
                                                curr_blob_image2_folder_path )\
                                            if "blob" in file ][0]
                curr_blob_image2_file_path = os.path.join ( curr_blob_image2_folder_path,
                                                        curr_blob_image2_file )
                
                print ( curr_blob_image2_file_path, '\n', next_blob_image2_file_path )

                swap_image_files ( file_path_1 = next_blob_image2_file_path,
                                    file_path_2 = curr_blob_image2_file_path ) 

    image_pair_folders_name = os.listdir ( dissimilar_folder_path )
    image_pair_folder_paths = [ 
        os.path.join ( dissimilar_folder_path, image_pair_folder )\
        for image_pair_folder in image_pair_folders_name ]

    for image_pair in image_pair_folder_paths:
        swapping_blobs_in_image_pair ( image_pair )
    print ( "Done :)" )

def split_into_train_test ():

    # Define paths
    dataset_dir = r'VisualOdometry\feature_extraction\dataset_for_model'
    train_dir = r'VisualOdometry\feature_extraction\dataset_for_model\train'
    test_dir = r'VisualOdometry\feature_extraction\dataset_for_model\test'

    # Create train and test directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Define class folders
    class_folders = os.listdir(dataset_dir)

    # Define train-test split ratio (e.g., 80-20)
    train_ratio = 0.8

    # Iterate through each class folder
    for folder in class_folders:
        class_subfolders = os.listdir(os.path.join(dataset_dir, folder))
        random.shuffle(class_subfolders)
        train_size = int(len(class_subfolders) * train_ratio)

        # Split folders into train and test sets
        train_subfolders = class_subfolders[:train_size]
        test_subfolders = class_subfolders[train_size:]

        # Move train folders
        for subfolder in train_subfolders:
            src = os.path.join(dataset_dir, folder, subfolder)
            dst = os.path.join(train_dir, folder, subfolder)
            shutil.move(src, dst)

        # Move test folders
        for subfolder in test_subfolders:
            src = os.path.join(dataset_dir, folder, subfolder)
            dst = os.path.join(test_dir, folder, subfolder)
            shutil.move(src, dst)

    print("Dataset split into train and test folders successfully.")

def rotate_image_in_image_pair_folder ( image_pair_folder : str, index : int ):
    rotation_transformation = [ cv2.ROTATE_180, 
                            cv2.ROTATE_90_CLOCKWISE,
                            cv2.ROTATE_90_COUNTERCLOCKWISE ]
    
    for image_folder in os.listdir ( image_pair_folder ):
        
        image_folder_path = os.path.join ( image_pair_folder, image_folder )
        
        for image_file in os.listdir ( image_folder_path):
            image_file_path = os.path.join ( image_folder_path, image_file )
            print ( image_file_path )
            image = cv2.imread ( image_file_path )
            cv2.imwrite (image_file_path, 
                        cv2.rotate (image,rotation_transformation[index]))

def augment_image_in_image_pair_folder ( image_pair_folder_path : str ):
    parent_dir = "\\".join(image_pair_folder_path.split ( "\\" ) [:-1])
    image_pair_folder_name = image_pair_folder_path.split("\\")[-1]

    for index in range ( 3 ):
        
        new_image_pair_folder_name = \
            image_pair_folder_name + "_" + str ( index )
        
        new_image_pair_folder_path = \
            os.path.join ( parent_dir, new_image_pair_folder_name )
        
        if not os.path.exists ( new_image_pair_folder_path ):
            shutil.copytree ( image_pair_folder_path,
                            new_image_pair_folder_path )
            rotate_image_in_image_pair_folder ( new_image_pair_folder_path, index )

def augment_images_of_folder ( folder_path ):
    for image_pair_folder in os.listdir ( folder_path ):

        image_pair_folder_path = os.path.join(folder_path,image_pair_folder)
        print ( image_pair_folder_path )
        augment_image_in_image_pair_folder ( image_pair_folder_path )
    
    print ( "Done :)" )

augment_images_of_folder ( r"VisualOdometry\feature_extraction\dataset_for_model\train\dissimilar" )