import shutil
import os

def swap_image_files(file_path_1, file_path_2):
    # Open both image files in binary read mode and read their contents
    with open(file_path_1, 'rb') as file1, open(file_path_2, 'rb') as file2:
        file1_content = file1.read()
        file2_content = file2.read()
    
    # Open both image files in binary write mode and swap their contents
    with open(file_path_1, 'wb') as file1, open(file_path_2, 'wb') as file2:
        file1.write(file2_content)
        file2.write(file1_content)

def swapping_blobs_in_image_pair ( image_pair_path : str):
        
    blob_folders_name =  os.listdir ( image_pair_path )
    
    for index, blob_folder in enumerate ( blob_folders_name ):
            
        if index < len ( blob_folders_name ) - 1:
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
            
            swap_image_files ( file_path_1 = next_blob_image2_file_path,
                                file_path_2 = curr_blob_image2_file_path )            
            
        else:
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

shutil.copytree ( r"VisualOdometry\feature_extraction\dataset\blobs",
                 r"VisualOdometry\feature_extraction\dataset\dissimilar" )

# swapping_blobs_in_image_pair ( 
#     r"VisualOdometry\feature_extraction\dataset\dissimilar\image_pair2" )
