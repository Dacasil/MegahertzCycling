import functions as func #Import the functions.py python file where all the import functions are defined

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import mrcfile # Work with mrc data
import tifffile as tiff # To read Tiff files

import os # To read an Folder and iterate over its files

from tqdm import tqdm

import cv2

from scipy.signal import chirp, find_peaks, peak_widths

###...............Functions...............###
# Calculate the radius of the main spots to use them on the mask
def calculate_radius(image, coord_list, radius):
    image = image.astype(np.uint8)
    shape_of_data = np.shape(image)
    #Define the meshgrid which returns two 2D arrays representing X and Y coordinates of all the points
    Y, X = np.ogrid[:shape_of_data[0], :shape_of_data[1]]
    mask = np.zeros((shape_of_data[0],shape_of_data[1]), dtype=bool)
    
    for coord in coord_list:
        mask1 = (X-coord[0])**2 + (Y-coord[1])**2 <= radius**2
        mask = mask | mask1
    
    image_masked = mask * image
    contours, _ = cv2.findContours(image_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Find the counters
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    maximum_rad = ((w/2)+(h/2))/2
    radius_of_object = []
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        radius = ((w/2)+(h/2))/2
        radius_of_object.append(radius)
    
    average_rad = ((np.array(radius_of_object)).sum())/(len(radius_of_object))
    return average_rad

###.................Data.................###
#Integrated Data
integrated_data = r'Data\Processed_Data\A13_processed\Integrated_images\Reprate_scan_raw_4F'
name_summed_intensity_file = '1_101kHz_5mW_drift_corrected_summed_intensity.tif'
'''
data_raw = tiff.imread(integrated_data+'/'+name_summed_intensity_file) #raw data
data_median_filtered = func.median_filter(data_raw) # median_filtered
shape_of_data = np.shape(data_raw)
plt.imshow(data_median_filtered, cmap='gray')
plt.show()'''

#Raw drift corrected data
test_data = r'Data\Processed_Data\A4_processed\Drift_corrected_images\RScan\0_0mW\0kHz_0mW_drift_corrected20230731_27166_TEM513_A4_DIFF40cm_0kHz_0mW_TaTe2_final.tif'
test_data_read = tiff.imread(test_data)
test_median_filtered = func.median_filter(test_data_read)
plt.imshow(test_data_read, cmap='gray')
plt.show()

# Find spot coordinates
base_coords = func.find_clustered_diffraction_spots(integrated_data, name_summed_intensity_file)
beam = base_coords[0]
main_coords = base_coords[1]
cdw_coords = base_coords[2]

###...............Programm...............###
# Radius of the integrated intensity images
Filtered_Image = func.gauss_filter_image(data_median_filtered)
print("###...Integrated intensity image spot radius...###")
print("Main spot radius:")
print(calculate_radius(Filtered_Image, main_coords, 100))
print("Cdw spot radius:")
print(calculate_radius(Filtered_Image, cdw_coords, 100))

# Radius of the single raw data image
Filtered_Image_raw = func.gauss_filter_image(test_median_filtered)
print("###...Single intensity image spot radius...###")
print(calculate_radius(Filtered_Image_raw, main_coords, 100))