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

def vibration_intensity(coords, drift_file_path):
    
    drift_data = []
    intensity = [] # Store the arrays which contain information about the change of intensity in the different spots
    
    for filename in os.listdir(drift_file_path):
        im = tiff.imread(drift_file_path + '/' + filename)
        drift_data.append(im)
    
    for i in range(len(drift_data)):
        
        #Get the data from the files
        file_1 = np.array(drift_data[i])
        all_intensity = file_1.sum() # The total intensity in the image
        
        shape_of_data = np.shape(file_1) # calculate shape (the shape of file_1 and file_2 are equal)
        
        #Calculate intensity of the coordinates in file_1:
        #fitted_coords_1 = func.fit_data(file_1, coords_of_interest, ROIrad_for_fit)
        intensity_1 = func.intensity_in_image(coords, 20, shape_of_data, file_1) # Use the intensity function
        normed_intensity = (intensity_1/all_intensity) *100
        intensity.append([normed_intensity,i])
        
    print("Done")
        
    return np.array(intensity)


def remove_words(text, words_to_remove):
    # Split the text into words
    words = text.split()

    # Remove the specified words
    cleaned_words = [word for word in words if word not in words_to_remove]

    # Reconstruct the cleaned text
    cleaned_text = ' '.join(cleaned_words)

    return cleaned_text


#Check whether the current script is being run as the main program or if it's being imported as a module into another script
if __name__ == '__main__':
    
    ###..........................................Data..........................................###

    drift_corrected_file_path = r'Data\Processed_Data\A4_processed\Drift_corrected_images\FScan254' # File path where the drift corrected images are stored

    #ATTENTION!!! The integrated images must have the same order as the drift_corrected data
    #Means: For a reprate scan: the right order of the reprate images; For a fluence scan: the right order of the fluence data
    #Convention here is starting with 0kHz(0mW) and going to 2033kHz (xmW) 
    integrated_data = r'Data\Processed_Data\A4_processed\Integrated_images\FScan254raw'
    zero_integrated_data = r'Data\Processed_Data\A4_processed\Integrated_images\FScan254raw'
    integrated_data_files = os.listdir(integrated_data)
    zero_data_name = "0_0kHz_0mW_drift_corrected_summed_intensity.tif"
    
    name_of_sample = "A4"
    name_std_intensity_txt = "standard_deviation_value_Fluence_Scan.txt"
    reprate_fluence_name = "Fluence_Scan"
    folder  = "7_5mW"
    number = 6   
    ###...............Constants...............###
    h_k_l = [(2,0,2), (-2,0,-2), (1,-3,1), (-1,3,-1), (-1,-3,-1), (1,3,1), (0,6,0), (0,-6,0), (2,6,2), (-2,-6,-2), (3,3,3), (-3,-3,-3), (4,0,4), (-4,0,-4), (3,-3,3),(-3,3,-3), (2,-6,2),(-2,6,-2)]
    
    
    
    ###......................................Programm......................................###
    
    chunks = []
    drift_data = [] # Save the drift data
    click_coords_array = [] # Save the clicked Coordinates
    drift_file_path = drift_corrected_file_path + "/" + folder # path to the drift corrected data
    len_drift_data = len(os.listdir(drift_file_path))
    print("Len drift data:")
    print(len_drift_data)
    print(integrated_data_files[number])
    
    #Find 0mW/kHz spot coordinates
    zero_khZ_spot_coords = func.find_clustered_diffraction_spots_by_clicking(zero_integrated_data , zero_data_name)
    cdw_zero  = zero_khZ_spot_coords[2] # CDW coordinates of the 0mW/0kHz data
    cdw_coords_and_false_spots_zero = zero_khZ_spot_coords[4]
    main = zero_khZ_spot_coords[1]
    cdw_unit = zero_khZ_spot_coords[5]
    main_unit = zero_khZ_spot_coords[6]
    
    #Find spots
    spot_coords = func.find_clustered_diffraction_spots_by_clicking(integrated_data, integrated_data_files[number])
    beam = spot_coords[0]
    main_coords = spot_coords[1]
    main_coords_lattice = spot_coords[4] # all main coords
    data_raw = tiff.imread(integrated_data+'/'+integrated_data_files[number]) # Data to determine the diffraction coordinates
    
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
        
        #plot the found spots to check
        for l in range(0,len(cdw)):
            plt.scatter(cdw[l][0], cdw[l][1], facecolors='none', edgecolors='b', s=60)
        for l in range(0,len(main_coords)):
            plt.scatter(main_coords[l][0], main_coords[l][1], facecolors='none', edgecolors='r', s=60)
        plt.imshow(data_raw,cmap='gray')
        plt.show()
    
    else:
        cdw = spot_coords[2]
        cdw_coords_and_false_spots = spot_coords[3] # coords for all cdw spots and false spots
    
    
    # Find the clicked coordinates
    for i in range(len(h_k_l)):
        print(h_k_l[i])
        clicked_coordinates = clicked_coord(integrated_data, integrated_data_files[number])
        click_coords_compared = func.compare_coord(clicked_coordinates, main_coords)
        print(click_coords_compared)
        click_coords_array.append(click_coords_compared)
    
    click_coords_array = np.array(click_coords_array)
    
    #Plot the spots to see if everything fits
    img =  tiff.imread(integrated_data + '/' + integrated_data_files[number]) # Read in the image
    img  = img.astype(np.uint8)
    for l in range(0,len(click_coords_array)):
        for i in range(len(click_coords_array[l])):
            plt.scatter(click_coords_array[l][i][0], click_coords_array[l][i][1], facecolors='none', edgecolors='r', s=60)
    plt.imshow(img,cmap='gray')
    plt.show()
            
    for j in range(len(click_coords_array)):
            chunks.append((click_coords_array[j], drift_file_path))
    
    print(len(chunks))
    
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=18) as pool:
        
        # Use pool.map to apply the function to the argument tuples in parallel
        results = pool.starmap(vibration_intensity, chunks)
        
        std_intensity= []
        time_steps = [time[1] for time in results[0]]
        
        for i in range(len(results)):
            intensity_main = [intensity[0] for intensity in results[i]]
            average_intensity_main = (np.array(intensity_main).sum())/(len_drift_data) # Average intensity per image
            average_intensity_main = round(average_intensity_main, 3)
            std_dev_main = np.std(intensity_main, ddof=1) # Standard deviation
            normed_std_dev_main = std_dev_main/average_intensity_main
            std_intensity.append([h_k_l[i], std_dev_main, average_intensity_main, normed_std_dev_main])
            
            
            ###..........Show the plot.............###
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
            plt.title("The intensity of the " + str(h_k_l[i]) + " spot")
            
            # Messdaten plotten
            plt.plot([np.min(np.array(time_steps)), np.max(np.array(time_steps))], [average_intensity_main, average_intensity_main], color='red', linewidth=1, label = 'average')
            plt.errorbar(time_steps, intensity_main, fmt= 'rs-', linewidth=1, ecolor="red", capsize=3, label = f"Average brag-spot intensity = {average_intensity_main} \nStandard derivation = {std_dev_main}")
            plt.plot(time_steps, intensity_main, 'r-')
            
            # Darstellung der Legende
            plt.legend(fancybox=False, loc="best")

            # Bild zuschneiden, abspeichern
            plt.tight_layout()
            plt.savefig(r'Data\Processed_Data' + '/' + name_of_sample + '_processed' + '/' +'Vibrations' + '/'+ reprate_fluence_name + '/' + folder + '/'+ str(i)+ '_'+ 'bragg_spots_' + folder + str(h_k_l[i]) + '.tif')
            plt.close()
    
    pool.close()
    pool.join()
    
    # Save the std deviation intensity
    with open(r'Data\Processed_Data' + '/' + name_of_sample + '_processed' + '/' +'Vibrations' + '/' + reprate_fluence_name + '/' + folder + '/' + name_std_intensity_txt , "w") as file:
        writer = csv.writer(file, delimiter = ',')
        writer.writerow(["spot_number", "std_dev_main", "average_main_intensity", "normed_std_dev"])
        writer.writerows(std_intensity)
        
        
