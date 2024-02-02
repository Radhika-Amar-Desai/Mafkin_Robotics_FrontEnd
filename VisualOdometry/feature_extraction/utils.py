import cv2

def show_image ( image : list , label = "Image" ):
    cv2.namedWindow( label , cv2.WINDOW_NORMAL)
    cv2.imshow ( label , image )
    cv2.resizeWindow( label , 800, 600) 
    cv2.waitKey ( 0 )
