import numpy as np
import cv2

def pad_image ( image : list, border_len = 1 ):
    """
        Pads image with 0s such that length and height
        of image increase by border_len.
    """
    return np.pad ( image , border_len, mode = "constant" )

def F1( intensity : int):
    return int ( intensity >= 0 )

def F3 ( k1 : int, k2 : int ):
    is_k1_non_negative = k1 >= 0
    is_k2_non_negative = k2 >= 0

    return ~(is_k1_non_negative ^ is_k2_non_negative) & 1

def labeled_coord_of_patch ( n : int, x : int , y : int ):
    """
        Returns coordinates (x,y) of image according to 
        label number n
    """
    try:
        labeled_coords = [(0,0), (0,1), (1,1), 
                          (1,0), (1,-1), (0,-1), 
                          (-1,-1), (-1,0), (-1,1) ]
        coord = labeled_coords [ n ]
        #print ( coord [0] + x , coord [1] + y )
        return ( coord [0] + x , coord [1] + y )
    except Exception as e:
        raise ValueError("Provide x and y between 0 and 8") from e

def LBP( image : list, x : int, y : int ):
    """
        Performs Local Binary Pattern.
        Returns binary descriptor for patch whose center coordinates are given by input x and y. 
    """
    #print ( x , y )
    return sum([(2 ** (i-1)) * \
    F1(image[labeled_coord_of_patch(i,x,y)[0]][labeled_coord_of_patch(i,x,y)[1]] \
       - image[labeled_coord_of_patch(0,x,y)[0]][labeled_coord_of_patch(0,x,y)[1]]) \
    for i in range ( 1 , 9 ) ])

def LNDP ( image : list, x : int, y : int ):
    """
        Performs Local Neighbourhood Pattern.
        Returns binary descriptor for patch whose center coordinates are given by input x and y. 
    """
    def calculate_k1 ( n : int ):
        m = n - 1
        coords_for_In = labeled_coord_of_patch ( n , x,  y )

        if n == 1:
            coords_for_I8 = labeled_coord_of_patch ( 8 , x , y ) 
            #print ( coords_for_I8 )
            return image [ coords_for_I8[0] ][ coords_for_I8[1] ] - image [ coords_for_In[0] ][ coords_for_In[1] ]
        elif 2 <= n <= 8:
            coords_for_Im = labeled_coord_of_patch ( m , x, y )
            #print(coords_for_Im, coords_for_In)
            return image [ coords_for_Im[0]][ coords_for_Im[1] ] - image [ coords_for_In[0]][ coords_for_In[1] ]

    def calculate_k2 ( n : int ):
        l = n + 1
        coords_for_In = labeled_coord_of_patch ( n , x,  y )
        if 1 <= n <= 7:
            coords_for_Il = labeled_coord_of_patch ( l , x, y )
            return image [ coords_for_Il[0] ][ coords_for_Il[1] ] - image [ coords_for_In[0] ][ coords_for_In[1] ]
        elif n == 8:
            coords_for_I1 = labeled_coord_of_patch ( 1 , x , y )
            return image [ coords_for_I1[0] ][ coords_for_I1[1] ] - image [ coords_for_In[0] ][ coords_for_In[1] ]
        
    #print ( [ ( calculate_k1 ( i ) , calculate_k2 ( i ) ) for i in range ( 1, 9 )] )
    
    return sum([ 2 ** (i-1) * F3( calculate_k1 ( i ) , calculate_k2 ( i ) ) for i in range ( 1, 9 )])

def get_feature_vector_of_image_using_local_pattern_descriptor ( image : list, LPD ):
    """
        Gives feature vector for the image obtained by specified local pattern descriptor like 
        LBP ( Local Binary Pattern ) and LNDP ( Local Neighbourhood Pattern ).
    """
    h , w = image.shape
    padded_image = pad_image ( image )
    LPD_of_each_pixel = [ LPD ( padded_image , 
                               x + 1, y + 1 ) \
                            for x in range ( h ) \
                            for y in range ( w ) ]
    #print ( LPD_of_each_pixel )
    hist, _ = np.histogram ( LPD_of_each_pixel, 
                            bins = 256, range = ( 0 , 256 ) )

    return np.array(hist)

def get_normalized_vec( data : list, min_val : int, max_val : int) -> list:
    """
    Perform Min-Max normalization on the input data.

    Parameters:
    - data: A numpy array or list containing the data to be normalized.

    Returns:
    - normalized_data: The normalized data.
    """

    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def get_normalized_feat_vect_of_image_using_LBP_LNDP ( image : list ):
    """
        Gives feature vector for the image obtained by concatenating histograms of image obtained by
        LBP (Local Binary Pattern) and LNDP (Local Neighbourhood Pattern).
    """
    feat_vec = get_feature_vector_of_image_using_local_pattern_descriptor ( image, LBP ) +\
        get_feature_vector_of_image_using_local_pattern_descriptor ( image, LNDP )

    return get_normalized_vec ( feat_vec, 0, 256 )
    
def get_dissimilarity_score_for_feature_vectors ( first_feature_vec, second_feature_vec ):
    """
        Gives similarity score between two feature vectors of images.
    """
    return sum(abs((first_feature_vec - second_feature_vec) / ( 1 + first_feature_vec + second_feature_vec )))
        
def get_dissimilarity_score_for_two_images ( first_image, second_image ):
    """
        Gives dissimilarity score between two images with the help of their feature vectors computed using LBP + LNDP
        technique.
    """
    # first_image = cv2.resize ( first_image, ( 50 , 50 ) )
    # second_image = cv2.resize ( second_image, ( 50 , 50 ) )

    if len ( first_image.shape ) == 3 :
        first_image = cv2.cvtColor( first_image, cv2.COLOR_BGR2GRAY )
    if len ( second_image.shape ) == 3 :
        second_image = cv2.cvtColor( second_image, cv2.COLOR_BGR2GRAY )

    return get_dissimilarity_score_for_feature_vectors ( 
                    get_normalized_feat_vect_of_image_using_LBP_LNDP( first_image ),
                    get_normalized_feat_vect_of_image_using_LBP_LNDP ( second_image ) )

first_image_path = \
    r"VisualOdometry\feature_extraction\dataset\blobs\image_pair1\blob0\image1\blob_at_1573_2187_image_1.jpg"
    #r"VisualOdometry\feature_extraction\dataset\blobs\image_pair2\blob0\image2\blob_at_1856_2732_image_2.jpg"   
    #r"VisualOdometry\feature_extraction\dataset\blobs\image_pair1\blob0\image1\blob_at_1573_2187_image_1.jpg"
second_image_path = \
    r"VisualOdometry\feature_extraction\dataset\blobs\image_pair1\blob0\image2\blob_at_2448_1924_image_2.jpg"
    #r"VisualOdometry\feature_extraction\dataset\blobs\image_pair1\blob1\image1\blob_at_1676_1577_image_1.jpg"
    #r"VisualOdometry\feature_extraction\dataset\blobs\image_pair2\blob2\image1\blob_at_2140_1980_image_1.jpg"
    #r"VisualOdometry\feature_extraction\dataset\blobs\image_pair2\blob4\image1\blob_at_2254_1441_image_1.jpg"
    #r"VisualOdometry\feature_extraction\dataset\images\image_pair1\image2.jpg"  
    #r"VisualOdometry\feature_extraction\dataset_for_model\test\dissimilar\image_pair3_blob11\image1\blob_at_1743_2346_image_1.jpg"

first_image = cv2.imread ( first_image_path , 0 )
second_image = cv2.imread ( second_image_path , 0 )

first_image = cv2.resize ( first_image, (128,128) )
second_image = cv2.resize ( second_image, (128,128) )

first_image = first_image.astype ( np.int64 )
second_image = second_image.astype ( np.int64 )

print ( get_normalized_feat_vect_of_image_using_LBP_LNDP ( first_image ).shape )

print ( get_dissimilarity_score_for_two_images ( first_image, second_image ) )