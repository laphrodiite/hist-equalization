import numpy as np
import sys
import math


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


def calculate_eq_transformations_of_regions(img_array: np.ndarray, region_len_h: int, region_len_w: int):

    # Initializing a dictionary that will have the first pixel in a contextual region
    # as keys and the region's transformation vector as values
    trans_dict = {}

    # Calculating the size of the image and the number of regions in every row and column
    H, W = np.shape(img_array)
    rows = int(H/region_len_h)
    columns = int(W/region_len_w)

    for i in range(columns):
        for j in range(rows):
            # For each region finds all pixels belonging to it and calls the function
            region_array = img_array[j*region_len_h:(j+1)*region_len_h-1, i*region_len_w:(i+1)*region_len_w-1]
            region_trans = get_equalization_transform_of_img(region_array)

            trans_dict[(j*region_len_h, i*region_len_w)] = region_trans
    
    return trans_dict


def perform_adaptive_hist_equalization(img_array: np.ndarray, region_len_h: int, region_len_w: int) -> np.ndarray:
    regions_dict = calculate_eq_transformations_of_regions(img_array, region_len_h, region_len_w)

    # Calculating size and initializing the adaptive image's array
    H, W = np.shape(img_array)
    adaptive = np.zeros((H,W))

    for i in range(H-1):
        for j in range(W-1):
            # Checking if the pixel is in the outer (green) area
            if i<=region_len_h/2 or i>=H-region_len_h/2 or j<=region_len_w/2 or j>=W-region_len_w/2:
                # Finding the number of the region the pixel belongs to
                region_h = int(np.floor(i/region_len_h)) 
                region_w = int(np.floor(j/region_len_w)) 
                region = (region_h * region_len_h, region_w * region_len_w)
                # Finding the transformation vector of the region and then the specific value for the pixel
                self_adaptive = regions_dict[region]
                adaptive[i, j] = self_adaptive[img_array[i,j]]
            else:
                # Calctlating the region left and up (minus, minus)
                    mm_region = (region_len_h*int(np.floor((i-region_len_h/2)/region_len_h)), region_len_w * int(np.floor((j-region_len_w/2)/region_len_w)))
                    mm_adaptive = regions_dict[mm_region]
                    mm_center = [mm_region[0]+int(region_len_h/2), mm_region[1]+int(region_len_w/2)]
    
                # Calculating the region left and down (plus, minus)
                    pm_region = (region_len_h*int(np.floor((i+region_len_h/2)/region_len_h)), region_len_w*int(np.floor((j-region_len_w/2)/region_len_w)))
                    pm_adaptive = regions_dict[pm_region]
                    
                # Calculating the region right and down (plus, plus)
                    pp_region = (region_len_h*int(np.floor((i+region_len_h/2)/region_len_h)), region_len_w*int(np.floor((j+region_len_w/2)/region_len_w)))
                    pp_adaptive = regions_dict[pp_region]
                    pp_center = [pp_region[0]+int(region_len_h/2), pp_region[1]+int(region_len_w/2)]
                
                # Calculating the region light and up (minus, plus)
                    mp_region = (region_len_h*int(np.floor((i-region_len_h/2)/region_len_h)), region_len_w*int(np.floor((j+region_len_w/2)/region_len_w)))
                    mp_adaptive = regions_dict[mp_region]
                    
                    a = (j - mm_center[1]) / (pp_center[1] - mm_center[1])
                    b = (i - mm_center[0]) / (pp_center[0] - mm_center[0])

                    adaptive[i, j] = (1 - a)*(1 - b)*mm_adaptive[img_array[i,j]] + (1-a)*b*pm_adaptive[img_array[i,j]] + a*(1-b)*mp_adaptive[img_array[i,j]] + a*b*pp_adaptive[img_array[i,j]]

    return adaptive

