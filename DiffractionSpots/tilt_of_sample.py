# Import Libraries
import numpy as np
import cv2
import tifffile as tiff # To read Tiff files
import functions as func #Import the functions.py python file where all the import functions are defined
import matplotlib.pyplot as plt
import os
from scipy import optimize # To make a linear fit
import csv

# Before starting change Gauss sigma 1 and 2 to (1,2) instead of (10,16)
# And change in click_coordinates the raw image into the gauss filtered image

###..................................Functions........................###
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

def intensity_for_tilt(coord, radius, shape_of_data, data):
    #Define the meshgrid which returns two 2D arrays representing X and Y coordinates of all the points
    Y, X = np.ogrid[:shape_of_data[0], :shape_of_data[1]]
    mask = np.zeros((shape_of_data[0],shape_of_data[1]), dtype=bool)
    
    mask1 = (X-coord[0])**2 + (Y-coord[1])**2 <= radius**2
    mask = mask | mask1
    
    masked_image = data *mask
    intensity = masked_image.sum()
    return intensity

def image_mask(coord_list, radius, shape_of_data, data):
    #Define the meshgrid which returns two 2D arrays representing X and Y coordinates of all the points
    Y, X = np.ogrid[:shape_of_data[0], :shape_of_data[1]]
    mask = np.zeros((shape_of_data[0],shape_of_data[1]), dtype=bool)
    
    for coord in coord_list:
        mask1 = (X-coord[0])**2 + (Y-coord[1])**2 <= radius**2
        mask = mask | mask1
    
    masked_image = data *mask
    return masked_image

###.................................Data.................................###
unexcited_data_path = r"Data\Processed_Data\A13_processed\Tilt\2033kHz\120mW"
unexcited_data_name = "0_TEM513_DIFF40cm_2033kHz_120mW_TaTe2_LT_final_tilt_data.tif"
excited_data_path = r"Data\Processed_Data\A13_processed\Tilt\2033kHz\120mW"
excited_data_name = "1_TEM513_DIFF40cm_2033kHz_120mW_TaTe2_LT_final_tilt_data.tif"

image_data_unexcited  = tiff.imread(unexcited_data_path+'/'+unexcited_data_name) # Get the image data of the unexcited data
image_data_excited = tiff.imread(excited_data_path +'/'+ excited_data_name) # Get the image data of the excited data
shape_of_data = np.shape(image_data_unexcited) # Shape of data

folder_path_for_images = r'Data\Processed_Data\A13_processed\Tilt\2033kHz' # Folder to save the images
folder_name = "Tilt_images"
unexcited_name = folder_name + '_' + "unexcited" + '120mW'
excited_name = folder_name + '_' + "excited" + '120mW'

###.......................................Program.......................................###

# Take the mass middle point of the Image and examine how it's changing
unexcited_spots = func.find_clustered_diffraction_spots_by_clicking(unexcited_data_path, unexcited_data_name)
main_spots_unexcited = unexcited_spots[1] # Bragg Spots
beam_unexcited = unexcited_spots[0] # beam coordinates
x_roi_beam_unexcited = beam_unexcited[0] - (beam_unexcited[0]-400)
y_roi_beam_unexcited = beam_unexcited[1] - (beam_unexcited[1]-400)
beam_unexcited = np.array(beam_unexcited, dtype='i')
ROI_unexcited = image_data_unexcited[beam_unexcited[1]-400:beam_unexcited[1]+401, beam_unexcited[0]-400:beam_unexcited[0]+401]

excited_spots = func.find_clustered_diffraction_spots_by_clicking(excited_data_path, excited_data_name) 
main_spots_excited = excited_spots[1]
beam_excited = excited_spots[0]
x_roi_beam_excited = beam_excited[0] - (beam_excited[0]-400)
y_roi_beam_excited = beam_excited[1] - (beam_excited[1]-400)
beam_excited = np.array(beam_excited, dtype='i')
ROI_excited = image_data_excited[beam_excited[1]-400:beam_excited[1]+401, beam_excited[0]-400:beam_excited[0]+401]


# Calculate center of mass
mass_center_unexcited = func.center_of_mass(image_data_unexcited)
x_mass_center_unexcited = mass_center_unexcited[0] - (beam_unexcited[0]-400)
y_mass_center_unexcited = mass_center_unexcited[1] - (beam_unexcited[1]-400)

mass_center_excited = func.center_of_mass(image_data_excited)
x_mass_center_excited = mass_center_excited[0] - (beam_excited[0]-400)
y_mass_center_excited = mass_center_excited[1] - (beam_excited[1]-400)

print("Center of mass of unexcited data")
print(mass_center_unexcited)
plt.scatter(beam_unexcited[0], beam_unexcited[1], marker ='o', s= 30 , facecolors='none', edgecolors = 'y', label = '(000)' )
plt.scatter(mass_center_unexcited[0], mass_center_unexcited[1], marker = 'x', s=20, color='r', label = 'Intensity center of mass')
plt.imshow(image_data_unexcited,cmap='gray')
plt.legend(fancybox=False, loc="best")
plt.savefig(folder_path_for_images + '/' + folder_name + '/' + unexcited_name + '.pdf')
plt.show()

plt.scatter(x_roi_beam_unexcited, y_roi_beam_unexcited, marker ='o', s= 20 , facecolors='none', edgecolors = 'y', label = '(000)' )
plt.scatter(x_mass_center_unexcited, y_mass_center_unexcited, marker = 'x', s=20, color='r', label = 'Intensity center of mass')
plt.imshow(ROI_unexcited,cmap='gray')
plt.legend(fancybox=False, loc="best")
plt.savefig(folder_path_for_images + '/' + folder_name + '/' + 'ROI_' + unexcited_name + '.pdf')
plt.show()

print("Center of mass of excited data")
print(mass_center_excited)
plt.scatter(beam_excited[0], beam_excited[1], marker ='o', s= 30 , facecolors='none', edgecolors = 'y', label = '(000)')
plt.scatter(mass_center_excited[0], mass_center_excited[1], marker = 'x', s=20, color='r', label = 'Intensity center of mass')
plt.imshow(image_data_excited,cmap='gray')
plt.legend(fancybox=False, loc="best")
plt.savefig(folder_path_for_images + '/' + folder_name + '/' + excited_name + '.pdf')
plt.show()

plt.scatter(x_roi_beam_excited, y_roi_beam_excited, marker ='o', s= 20 , facecolors='none', edgecolors = 'y', label = '(000)')
plt.scatter(x_mass_center_excited, y_mass_center_excited, marker = 'x', s=20, color='r', label = 'Intensity center of mass')
plt.imshow(ROI_excited,cmap='gray')
plt.legend(fancybox=False, loc="best")
plt.savefig(folder_path_for_images + '/' + folder_name + '/' + 'ROI_' + excited_name + '.pdf')
plt.show()

mass_center_unexcited = [x_mass_center_unexcited, y_mass_center_unexcited]
mass_center_excited = [x_mass_center_excited, y_mass_center_excited]

distance = func.calculate_distance(mass_center_unexcited, mass_center_excited)
print(distance)