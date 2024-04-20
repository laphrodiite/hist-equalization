from Assignment1_code import get_equalization_transform_of_img
from Assignment1_code import perform_global_hist_equalization
from Assignment1_code import calculate_eq_transformations_of_regions
from Assignment1_code import perform_adaptive_hist_equalization
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def import_image(filename):
    img = Image.open(fp=filename)
    bw_img = img.convert("L")
    img_array = np.array(bw_img)
    
    return img_array, bw_img

def plot_histograms(old, new):
    plt.figure(1)
    plt.title("Grayscale Image Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.hist(old)

    plt.figure(2)
    plt.title("Improved Grayscale Image Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.hist(new)
    
    plt.show()

def main():
    
    """
    img_array, bw_img = import_image("image1.png")
    
    final_array = perform_global_hist_equalization(img_array)

    bw_img.show()
    final_img = Image.fromarray(final_array)
    final_img.show()

    plot_histograms(img_array, final_array)
    """
    
    img_array, bw_img = import_image("image1.png")
    
    final_array = perform_adaptive_hist_equalization(img_array, 36, 48)

    bw_img.show()
    final_img = Image.fromarray(final_array)
    final_img.show()

    plot_histograms(img_array, final_array)    
    
    


if __name__ == "__main__":
	main()

