import cv2
import numpy as np
import torch

def show_image ( image : list , label = "Image" ):
    cv2.namedWindow( label , cv2.WINDOW_NORMAL)
    cv2.imshow ( label , image )
    cv2.resizeWindow( label , 800, 600) 
    cv2.waitKey ( 0 )

def euclidean_distance(tensor1, tensor2):
    # Compute element-wise squared difference
    # tensor1 = np.array ( tensor1 )
    # tensor2 = np.array ( tensor2 )

    tensor1 = tensor1.detach().numpy()
    tensor2 = tensor2.detach().numpy()

    squared_diff = (tensor1 - tensor2) ** 2

    # Sum the squared differences along appropriate dimensions
    sum_squared_diff = np.sum(squared_diff)

    # Take the square root of the sum of squared differences
    distance = np.sqrt(sum_squared_diff)

    return distance

# Example tensors
tensor1 = torch.randn ( 1 , 10 )
tensor2 = torch.randn ( 1 , 10 )

# #Calculate the Euclidean distance
# distance = euclidean_distance(tensor1, tensor2)
# print("Euclidean distance:", distance)