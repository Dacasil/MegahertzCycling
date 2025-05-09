#Basics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import mrcfile # work with data

from scipy.ndimage import gaussian_filter # Implement a gaussian filter
from skimage.feature.peak import peak_local_max # Find local maxima in the image

from IPython.display import clear_output # For the progressbar 


import itertools # Find the hexagon pattern in the clusters
import math # "..."

from itertools import product # Use for unit_cell_2d function

import cv2 # To read and process data
import os # To read an Folder and iterate over its files
import csv

import functions as func #Import the functions.py python file where all the import functions are defined

import tifffile as tiff # To read Tiff files




###....................................................................Program....................................................................###


#Enter Data for the processing
intensity_file_path = r'Data\Processed_Data\A7_processed\Intensity_of_integrated_images\Fluence_Scan_1016_A7' # Create file to save the intensities
path_of_data_raw = r'Data\Processed_Data\A7_processed\Integrated_images\FScan1016raw' # Enter path of the folder of which one wants to get the spot coordinates
find_coords_file = os.listdir(path_of_data_raw) # Need the same order as the final summed intensity files
path_of_data = r'Data\Processed_Data\A7_processed\Integrated_images\FScan1016' # Enter path of the folder of which one wants to get the Intensity
zero_data_name = "0_0kHz_0mW_drift_corrected_summed_intensity.tif" # 0mW/0kHz integrated data folder name

radius_main = 50 # Radius for the main spot surrounding circle
radius_cdw = 15 # Radius for the cdw spot surrounding circle

main_spot_intensity = [] # Array for the main spot intensities
cdw_spot_intensity = [] # Array for the cdw spot intensities
background_intensity = []
name_array = []

name_main_intensity_txt = 'main_spot_intensities_Fluence_Scan_1016.txt' # Name of the txt file where the intensity data is stored
name_cdw_intensity_txt = 'cdw_spot_intensities_Fluence_Scan_1016.txt' # "..."
name_background_intensity_txt = 'background_intensities_Fluence_Scan_1016.txt' # "..."

array_for_x_axis_reprate_scan = [0,677,1016,2033] # Reprates for the reprate scan in [kHz]
array_for_x_axis_fluence_scan = [0,5,10,15,17,20.5,25.5] # fluences for the fluence scan in [mW]
reprate_fluence_array = array_for_x_axis_fluence_scan #choose which array to take
number = -1


###...................Get drift corrected images and create a summed image...................###

sample_name  = 'A7' # Name of the sample
reprate_folder_name = 'FScan1016'
pth = 'Data/Processed_Data/'+ sample_name + '_processed/Drift_corrected_images/' + reprate_folder_name
drift_folder_file_number = []
drift_corrected_files = os.listdir(pth)


# Loop to sum up the drift corrected intensity
for drift_folder in drift_corrected_files:

    fluence_name = drift_folder
    #Get the drift corrected images in a numpy array
    im_drift_arrays = []
    pth = 'Data/Processed_Data/'+ sample_name + '_processed/Drift_corrected_images/' + reprate_folder_name +'/'+fluence_name
    len_of_data = len(os.listdir(pth)) # calculate the length of the drift_correction image array
    drift_folder_file_number.append(len_of_data) # save the number of the data to norm the average intensity of the image
    
    #Load Images into arrays
    for file_names in os.listdir(pth):
        im = tiff.imread(pth+'/'+file_names) 
        im_drift_arrays.append(im)

    # Integrate over all the images
    combined_intensity_not_normalized = np.sum(im_drift_arrays, axis=0)
    
    tiff.imwrite('Data/Processed_Data/'+ sample_name +'_processed/Integrated_images/'+ reprate_folder_name + '/'+ fluence_name + '_' +'drift_corrected_summed_intensity'+'.tif', combined_intensity_not_normalized)

# Find the spot coordinates of 0mW/0kHz
zero_khZ_spot_coords = func.find_clustered_diffraction_spots_by_clicking(path_of_data_raw , zero_data_name)
cdw_unit = zero_khZ_spot_coords[5]
main_unit = zero_khZ_spot_coords[6]

# Iteration over the drift corrected summed intensity data
for summed_intensity in os.listdir(path_of_data):
    print(summed_intensity)
    number +=1
    data_raw = tiff.imread(path_of_data+'/'+summed_intensity) #raw data
    
    data_median_filtered = func.median_filter(data_raw) # median_filtered
    main_shape = np.shape(data_raw) # Shape of the data
    
    #Now find and cluster the diffraction spots
    # Spot_coords returns: (1) Beam coordinates, (2) Main spots coordinates, (3) CDW spots coordinates, (4) CDW  coordinates and false coordinates, (5) All main coordinates
    spot_coords = func.find_clustered_diffraction_spots_by_clicking(path_of_data_raw , find_coords_file[number])
    beam = spot_coords[0]
    main = spot_coords[1]
    main_coords_lattice = spot_coords[4]
    
    
    false_spots_mask = spot_coords[7]
    
    #Filter out if data has found cdw spots
    if np.shape(spot_coords[2])[0] == 1:
        a_cdw = cdw_unit[0] # Get cdw unit vectors
        b_cdw = cdw_unit[1]
        a_main = main_unit[0] # Get main unit vectors
        b_main = main_unit[1]
        NxMain = 10
        NyMain = 7
        NxCDW = 10
        NyCDW = 10
        main_latt_coord = func.unit_cell_2D(a_main, b_main, beam, NxMain, NyMain) #Grid of main lattice spots
        cdw_latt_coord_raw = func.unit_cell_2D(a_cdw, b_cdw, beam, NxCDW, NyCDW) # Get cdw unit vector lattice
        
        
        #Currently the cdw lattice coordinates contain also the main lattice points -> so one wants to filter out the main lattice spots 
        mainSpots_in_cdw_latt = []
        minimum = float('inf')
        rval = 50
        for i in range(len(cdw_latt_coord_raw)):
                for j in range(len(main_latt_coord)):
                        dis = func.calculate_distance(cdw_latt_coord_raw[i], main_latt_coord[j])
                        if dis < minimum:
                                spot = cdw_latt_coord_raw[i]
                                minimum = dis
                        else: continue
                        
                if minimum < rval:
                        minimum = float('inf')
                        mainSpots_in_cdw_latt.append(spot)
                else: minimum = float('inf')
        
        #Now Find the cdw lattice spots by removing the "main lattice spots" (mainSpots_in_cdw_latt) from the cdw spots array
        #Do this with a mask
        cdw_latt_coord = []
        values_to_remove = np.intersect1d(mainSpots_in_cdw_latt,cdw_latt_coord_raw)
        
        mask = np.isin(cdw_latt_coord_raw, values_to_remove)

        for i in range(len(cdw_latt_coord_raw)):
            if (mask[i][0] & mask[i][1]) == False:
                cdw_latt_coord.append(cdw_latt_coord_raw[i])
                
        
        cdw_latt_coord = np.array(cdw_latt_coord)
        cdw = func.is_within_interval(cdw_latt_coord) # The cdw coordinates
        cdw_coords_and_false_spots = spot_coords[3]
        combined_coords = [] # Coords to get background intensity
        combined_coords.append(main_coords_lattice)
        combined_coords.append(cdw_coords_and_false_spots)
        
        #plot the found spots to check
        for l in range(0,len(cdw)):
            plt.scatter(cdw[l][0], cdw[l][1], facecolors='none', edgecolors='b', s=60)
        for l in range(0,len(main)):
            plt.scatter(main[l][0], main[l][1], facecolors='none', edgecolors='r', s=60)
        plt.imshow(data_raw,cmap='gray')
        plt.show()
        
    else:
        cdw = spot_coords[2]
        cdw_coords_and_false_spots = spot_coords[3] # coords for all cdw spots and false spots
        
        
        # all main coords
        combined_coords = [] # To insert in the background intensity function
        combined_coords.append(main_coords_lattice)
        combined_coords.append(cdw_coords_and_false_spots)
    
    
    all_intensity = data_raw.sum()
    
    #Get the average intensity of the diffraction spots
    #Do NOT forget to change the folder: reprate -> fluence or fluence -> reprate
    intensity_main = func.intensity_in_image(main, radius_main, main_shape, data_raw) # Main spots
    intensity_main = intensity_main/(drift_folder_file_number[number]) # Norm the main spot intensity on the number of recorded images
    
    intensity_cdw = func.intensity_in_image(cdw, radius_cdw, main_shape, data_raw) # CDW spots
    intensity_cdw = intensity_cdw/(drift_folder_file_number[number])
    
    intensity_background = func.background_intensity_in_image(combined_coords, radius_main, radius_cdw, main_shape, data_raw)[0]
    intensity_background = intensity_background/(drift_folder_file_number[number])
    
    intensity_all = all_intensity/(drift_folder_file_number[number]) # Norm the hole intensity on "..."
    
    intensity_main_normalised = (intensity_main/intensity_all)*100 # Normalise the main spot intensity on the total counted electron number
    main_spot_intensity.append([summed_intensity, reprate_fluence_array[number], intensity_main_normalised])
    
    intensity_cdw_normalised = (intensity_cdw/intensity_all)*100
    cdw_spot_intensity.append([summed_intensity, reprate_fluence_array[number], intensity_cdw_normalised])
    
    intensity_background_normalised = (intensity_background/intensity_all)*100
    background_intensity.append([summed_intensity, reprate_fluence_array[number], intensity_background_normalised])
    
    
    
    '''print(intensity_main)
    print(intensity_cdw)
    print(intensity_background)'''

'''
# Arrays for the diffraction spot intensities
main_intensity = []
cdw_intensity = []

# Get the Intensity value
for i in range(len(main_spot_intensity)):
    main_intensity.append(main_spot_intensity[i][2])
for i in range(len(cdw_spot_intensity)):
    cdw_intensity.append(cdw_spot_intensity[i][2])
'''

# Save the Summed and averaged intensity
with open(intensity_file_path + '/' + name_main_intensity_txt, "w") as file:
    writer = csv.writer(file, delimiter = ',')
    writer.writerow(["Info", "x_Axis", "Average Intensity"])
    writer.writerows(main_spot_intensity)

with open(intensity_file_path + '/' + name_cdw_intensity_txt, "w") as file:
    writer = csv.writer(file, delimiter = ',')
    writer.writerow(["Info", "x_axis", "Average Intensity"])
    writer.writerows(cdw_spot_intensity)
    
with open(intensity_file_path + '/' + name_background_intensity_txt, "w") as file:
    writer = csv.writer(file, delimiter = ',')
    writer.writerow(["Info", "x_axis", "Average Intensity"])
    writer.writerows(background_intensity)
