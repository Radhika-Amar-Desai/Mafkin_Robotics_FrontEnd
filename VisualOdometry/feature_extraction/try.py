import cv2
import os
import shutil

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

