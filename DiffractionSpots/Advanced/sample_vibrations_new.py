# Import Libraries
import numpy as np
import cv2
import tifffile as tiff # To read Tiff files
import functions as func #Import the functions.py python file where all the import functions are defined
import matplotlib.pyplot as plt
import os
from scipy import optimize # To make a linear fit
import csv

import multiprocessing
from tqdm import tqdm
import math
from PIL import Image

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

def vibration_intensity(mask, drift_file_path):
    
    drift_data = []
    intensity = [] # Store the arrays which contain information about the change of intensity in the different spots
    
    for filename in os.listdir(drift_file_path):
        im = tiff.imread(drift_file_path + '/' + filename)
        drift_data.append(im)
    
    for i in range(len(drift_data)):
        
        #Get the data from the files
        file_1 = np.array(drift_data[i])
        all_intensity = file_1.sum() # The total intensity in the image
        
        #Calculate intensity in the masked image:
        masked_image = mask * im
        intensity_1 = np.array(masked_image).sum()
        normed_intensity = (intensity_1/all_intensity) *100
        intensity.append([normed_intensity,i])
        #print(intensity)
        
    return np.array(intensity)

def remove_words(text, words_to_remove):
    # Split the text into words
    words = text.split()

    # Remove the specified words
    cleaned_words = [word for word in words if word not in words_to_remove]

    # Reconstruct the cleaned text
    cleaned_text = ' '.join(cleaned_words)

    return cleaned_text


###..........................................Data..........................................###

drift_corrected_file_path = r'Data\Processed_Data\A15_processed\Drift_corrected_images\Reprate_Scan' # File path where the drift corrected images are stored
integrated_data = r'Data\Processed_Data\A15_processed\Integrated_images\Reprate_scan_raw' # Must have the same order as the drift corrected images
integrated_data_files = os.listdir(integrated_data)
folder = "1_0mW"
name_of_sample = "A15"
reprate_fluence_name = "Reprate"
number = 0  # number to get the intgrated data fittning to the drift corrected files

###......................................Programm......................................###
drift_data = [] # Save the drift data
std_intensity = []

data_raw = tiff.imread(integrated_data+'/'+integrated_data_files[number]) #raw data
shape_of_data = np.shape(data_raw)

drift_file_path = drift_corrected_file_path + "/" + folder # path to the drift corrected data
len_drift_data = len(os.listdir(drift_file_path))

data_median_filtered = func.median_filter(data_raw) # median_filtered
FilteredImage = func.gauss_filter_image(data_median_filtered,GaussFilterSigma1=10,GaussFilterSigma2=16)
unclustered_intensity_spots = func.localization_intensity_spots(data_median_filtered,data_raw)

beam = func.find_beam_coord(FilteredImage,unclustered_intensity_spots)

# Show the found beam coordinates
plt.scatter(beam[0], beam[1], facecolors='none', edgecolors='r', s=60)
plt.imshow(FilteredImage, cmap='gray')
plt.show()

#Make a choice about the accuracy of the found spots
choice = input("Do you want to continue? (y/n): ")

if choice.lower() == "y":
    beam_coordinate = beam

# If it doesn't found the right spots
elif choice.lower() == "n":
    print("Click Beam") # choose beam coordinates by clicking
    beam = clicked_coord(integrated_data, data_raw)
    beam = np.array(beam[0])
    beam_coordinate = func.fit_data(data_raw, beam, 40)
    # Plot th new beam coordinates
    plt.scatter(beam_coordinate[0], beam_coordinate[1], facecolors='none', edgecolors='r', s=60)
    plt.imshow(FilteredImage, cmap='gray')
    plt.show()

# Create the masks
mask_array = []
#Define the meshgrid which returns two 2D arrays representing X and Y coordinates of all the points
Y, X = np.ogrid[:shape_of_data[0], :shape_of_data[1]]
mask = np.zeros((shape_of_data[0],shape_of_data[1]), dtype=bool) # Define the mask

# Mask1: Circular Mask for the first bragg spots
radius1 = 450
mask1 = (X-beam_coordinate[0])**2 + (Y-beam_coordinate[1])**2 <= radius1**2
mask_array.append(mask1)
mask1_image = FilteredImage * mask1
plt.imshow(mask1_image, cmap='gray')
plt.show()

# Mask3: Circular Mask for spots on the edge
radius3 = 800
mask_outside = (X-beam_coordinate[0])**2 + (Y-beam_coordinate[1])**2 <= radius3**2
mask3= np.logical_not(mask_outside)
mask_array.append(mask3)
mask3_image = FilteredImage * mask3
plt.imshow(mask3_image, cmap='gray')
plt.show()


# Mask2: Spots in the middle
mask_the_inner = np.logical_not(mask1)
mask2 = mask_outside & mask_the_inner
mask_array.append(mask2)
mask2_image = FilteredImage * mask2
plt.imshow(mask2_image, cmap='gray')
plt.show()

vibration = []
vibration_name = ["vibration_1","vibration_2","vibration_3"]
vibration_1 = vibration_intensity(mask1, drift_file_path) # vibration of the inner part
vibration.append(vibration_1)
vibration_2 = vibration_intensity(mask3, drift_file_path) # vibration of the middle part
vibration.append(vibration_2)
vibration_3 = vibration_intensity(mask2, drift_file_path) # vibration on the edge
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
    plt.savefig(r'Data\Processed_Data' + '/' + name_of_sample + '_processed' + '/' +'Vibrations_simple' + '/'+ reprate_fluence_name + '/' + folder + '/'+ vibration_name[i] + '.tif')
    plt.close()

# Save the std deviation intensity
with open(r'Data\Processed_Data' + '/' + name_of_sample + '_processed' + '/' +'Vibrations_simple' + '/' + reprate_fluence_name + '/' + folder + '/' + "standard_intensity.txt", "w") as file:
    writer = csv.writer(file, delimiter = ',')
    writer.writerow(["Vibration_name", "average_main_intensity", "normed_std_dev", "std_dev"])
    writer.writerows(std_intensity)