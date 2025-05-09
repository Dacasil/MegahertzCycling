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

import multiprocessing

from tqdm import tqdm

# Function to find searched coordinates in an image by clicking on them
def clicked_coord(path, name):
    coords_of_interest = [] # save coords of interest
    #Define a function to display the coordinates of
    #of the points clicked on the image
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f'({x},{y})')
            coords_of_interest.append((x,y)) # Safe coord data
        return coords_of_interest
        

    # read the input image
    img =  tiff.imread(path + '/' + name) # Read in the image
    img  = img.astype(np.uint8)

    # create a window
    cv2.namedWindow('Point Coordinates', cv2.WINDOW_NORMAL) # Defines the opened image window
    cv2.resizeWindow('Point Coordinates', 1000, 1000)

    # bind the callback function to window
    cv2.setMouseCallback('Point Coordinates', click_event)
    coords = click_event
    # display the image
    while True:
        cv2.imshow('Point Coordinates',img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    return coords_of_interest

def vibration_intensity(main_spots, cdw_spots, main_rad, cdw_rad, drift_file_path):
    
    drift_data = []
    normed_intensity_list = []
    i_list = []
    
    for filename in os.listdir(drift_file_path):
        im = tiff.imread(drift_file_path + '/' + filename)
        drift_data.append(im)
    
    for i in tqdm(range(len(drift_data))):
        
        #Get the data from the files
        file = np.array(drift_data[i])
        all_intensity = file.sum() # The total intensity in the image
        shape_of_data = np.shape(file)
        
        #Calculate intensity in the masked image:
        intensity_main = func.intensity_in_image(main_spots, main_rad, shape_of_data, file) # Use the intensity function
        intensity_cdw = func.intensity_in_image(cdw_spots, cdw_rad, shape_of_data, file) # Use the intensity function
        intensity = intensity_main + intensity_cdw
        normed_intensity = (intensity/all_intensity) *100
        
        # Append the data to separate lists
        normed_intensity_list.append(normed_intensity)
        i_list.append(i)
    
    # Convert the lists to NumPy arrays
    normed_intensity_array = np.array(normed_intensity_list)
    i_array = np.array(i_list)

    # Stack the arrays side by side
    intensity = np.column_stack((normed_intensity_array, i_array))
    
    return np.array(intensity)

def remove_words(text, words_to_remove):
    # Split the text into words
    words = text.split()

    # Remove the specified words
    cleaned_words = [word for word in words if word not in words_to_remove]

    # Reconstruct the cleaned text
    cleaned_text = ' '.join(cleaned_words)

    return cleaned_text

###....................................................................Program....................................................................###

#Enter Data for the processing
path_of_data_raw = r'Data\Processed_Data\A15_processed\Integrated_images\Fluence_scan_raw' # Enter path of the folder of which one wants to get the spot coordinates
drift_corrected_file_path = r'Data\Processed_Data\A15_processed\Drift_corrected_images\Fluence_Scan' # File path where the drift corrected images are stored
integrated_data = r'Data\Processed_Data\A15_processed\Integrated_images\Fluence_scan_raw' # Must have the same order as the drift corrected images
integrated_data_files = os.listdir(integrated_data)
zero_data_name = "1_2033kHz_0mW_drift_corrected_summed_intensity.tif" # 0mW/0kHz integrated data folder name

radius_main = 50 # Radius for the main spot surrounding circle
radius_cdw = 15 # Radius for the cdw spot surrounding circle

folder = "2_100mW"
name_of_sample = "A15"
reprate_fluence_name = "Fluence"
numbers = 1  # number to get the intgrated data fittning to the drift corrected files


###..................................Search for the spot coordinates..................................###
# Find the spot coordinates of 0mW/0kHz
zero_khZ_spot_coords = func.find_clustered_diffraction_spots_by_clicking(path_of_data_raw , zero_data_name)
cdw_unit = zero_khZ_spot_coords[5]
main_unit = zero_khZ_spot_coords[6]

###....................Get the spot coordinates....................###

data_raw = tiff.imread(integrated_data+'/'+integrated_data_files[numbers]) #raw data
drift_file_path = drift_corrected_file_path + "/" + folder # path to the drift corrected data
len_drift_data = len(os.listdir(drift_file_path)) # Len of the drift data

data_median_filtered = func.median_filter(data_raw) # median_filtered
FilteredImage = func.gauss_filter_image(data_median_filtered,GaussFilterSigma1=10,GaussFilterSigma2=16)
main_shape = np.shape(data_raw) # Shape of the data

#Now find and cluster the diffraction spots
# Spot_coords returns: (1) Beam coordinates, (2) Main spots coordinates, (3) CDW spots coordinates, (4) CDW  coordinates and false coordinates, (5) All main coordinates
spot_coords = func.find_clustered_diffraction_spots_by_clicking(path_of_data_raw , integrated_data_files[numbers])
beam = spot_coords[0]
main = spot_coords[1]
main_coords_lattice = spot_coords[4]

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
    combined_coords.append(main)
    combined_coords.append(cdw)
    
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
    combined_coords.append(main)
    combined_coords.append(cdw)


###.........................Cluster spot coordinates.........................###
main_spots = combined_coords[0]
cdw_spots = combined_coords[1]
radius1 = 450
radius2 = 800

main_spots_1 =[]
main_spots_2 = []
main_spots_3 = []

cdw_spots_1 =[]
cdw_spots_2 = []
cdw_spots_3 = []


for main in main_spots:
    difference = np.sqrt((beam[0]-main[0])**2 + (beam[1]-main[1])**2)
    if difference <= 450: main_spots_1.append(main)
    if difference > 450 and difference <=800: main_spots_2.append(main)
    if difference >= 800: main_spots_3.append(main)

for cdw in cdw_spots:
    difference = np.sqrt((beam[0]-cdw[0])**2 + (beam[1]-cdw[1])**2)
    if difference <= 450: cdw_spots_1.append(cdw)
    if difference > 450 and difference <=800: cdw_spots_2.append(cdw)
    if difference >= 800: cdw_spots_3.append(cdw)


for l in range(0,len(main_spots_1)):
    plt.scatter(main_spots_1[l][0], main_spots_1[l][1], facecolors='none', edgecolors='b', s=60)
for l in range(0,len(cdw_spots_1)):
    plt.scatter(cdw_spots_1[l][0], cdw_spots_1[l][1], facecolors='none', edgecolors='b', s=30)
plt.imshow(FilteredImage,cmap='gray')
plt.show()

for l in range(0,len(main_spots_2)):
    plt.scatter(main_spots_2[l][0], main_spots_2[l][1], facecolors='none', edgecolors='r', s=60)
for l in range(0,len(cdw_spots_2)):
    plt.scatter(cdw_spots_2[l][0], cdw_spots_2[l][1], facecolors='none', edgecolors='r', s=30)
plt.imshow(FilteredImage,cmap='gray')
plt.show()

for l in range(0,len(main_spots_3)):
    plt.scatter(main_spots_3[l][0], main_spots_3[l][1], facecolors='none', edgecolors='y', s=60)
for l in range(0,len(cdw_spots_3)):
    plt.scatter(cdw_spots_3[l][0], cdw_spots_3[l][1], facecolors='none', edgecolors='y', s=30)
plt.imshow(FilteredImage,cmap='gray')
plt.show()


###..............Calculate the standard derivation................###
std_intensity = []
vibration = []
vibration_name = ["vibration_1","vibration_2","vibration_3"]
vibration_1 = vibration_intensity(main_spots_1, cdw_spots_1, radius_main, radius_cdw, drift_file_path)
print(len(vibration_1))
vibration.append(vibration_1)
vibration_2 = vibration_intensity(main_spots_2, cdw_spots_2, radius_main, radius_cdw, drift_file_path)
vibration.append(vibration_2)
vibration_3 = vibration_intensity(main_spots_3, cdw_spots_3, radius_main, radius_cdw, drift_file_path)
vibration.append(vibration_3)

for i in tqdm(range(3)):
    intensity_main_1 = [intensity[0] for intensity in vibration[i]]
    time_steps = [time[1] for time in vibration[i]]
    average_intensity_main_1 = (np.array(intensity_main_1).sum())/(len_drift_data) # Average intensity per image
    average_intensity_main_1 = round(average_intensity_main_1, 3)
    std_dev_main_1 = np.std(intensity_main_1, ddof=1) # Standard deviation
    normed_std_dev_main_1 = std_dev_main_1/average_intensity_main_1
    normed_std_dev_main_1 = round(normed_std_dev_main_1,3)
    std_intensity.append([vibration_name[i], average_intensity_main_1, normed_std_dev_main_1, std_dev_main_1])

    # figure object erzeugen
    plt.figure(figsize=(10,6))

    #Gridlines erzeugen (wie man sieht, gibt es zig Möglichkeiten diese zu Zeichnen)
    plt.minorticks_on()
    plt.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.6)
    plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.3)

    # Achsenbeschriftungen
    plt.xlabel("Time in [s]", fontsize="14")
    plt.ylabel("Intensity in [%]", fontsize="14")
    
    # Titel für den Plot
    
    # Messdaten plotten
    plt.plot([np.min(np.array(time_steps)), np.max(np.array(time_steps))], [average_intensity_main_1, average_intensity_main_1], color='red', linewidth=1, label = 'average')
    plt.errorbar(time_steps, intensity_main_1, fmt= 'rs-', linewidth=1, ecolor="red", capsize=3, label = f"Average brag-spot intensity = {average_intensity_main_1} \nStandard derivation = {normed_std_dev_main_1}")
    
    # Darstellung der Legende
    plt.legend(fancybox=False, loc="best")

    # Bild zuschneiden, abspeichern
    plt.tight_layout()
    plt.savefig(r'Data\Processed_Data' + '/' + name_of_sample + '_processed' + '/' +'Vibrations' + '/'+ reprate_fluence_name + '/' + folder + '/'+ vibration_name[i] + '.tif')
    plt.close()

# Save the std deviation intensity
with open(r'Data\Processed_Data' + '/' + name_of_sample + '_processed' + '/' +'Vibrations' + '/' + reprate_fluence_name + '/' + folder + '/' + "standard_intensity.txt", "w") as file:
    writer = csv.writer(file, delimiter = ',')
    writer.writerow(["Vibration_name", "average_main_intensity", "normed_std_dev", "std_dev"])
    writer.writerows(std_intensity)