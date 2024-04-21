import numpy as np
import matplotlib.pyplot as plt


def get_equalization_transform_of_img(img_array: np.ndarray) -> np.ndarray:
    """
    Input: 2D numpy array, represents a 8-bit grayscale image
    Output: 1D numpy array of L elements

    L: number of possible values in the array
    """
    # Calculate the histogram and pixels of the original image
    H, W = np.shape(img_array)
    pixels = H*W

    
    histogram = np.zeros(256)
    for i in range(H):
        for j in range(W):
            histogram[img_array[i,j]] += 1
        

    # Calculating probability
    prob = np.zeros(256, dtype= float)
        
    for i in range(256):
        prob[i] = histogram[i] / pixels
    

    # Calculating the v vector
    v = [prob[0]]
    for k in range(1, len(prob)):
        v.append(v[k-1]+prob[k])
    L = len(v)


    # Calculating the y vector
    y = []
    for k in range(L):
        y.append(round((L-1)*(v[k]-v[0])/(1-v[0])))
    
    return y
    

def perform_global_hist_equalization(img_array: np.ndarray,) -> np.ndarray:

    # Input: 2D numpy array that is the original image
    # Output: 2D numpy array that is the equalized image

    T = get_equalization_transform_of_img(img_array)
    
    H, W = np.shape(img_array)

    y = np.zeros((H,W))

    # Calculating new values based on the equalization transform
    for i in range(H):
        for j in range(W):
            y[i,j] = T[img_array[i,j]]

    return y


