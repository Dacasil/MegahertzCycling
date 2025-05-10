import functions as func # Import the functions.py python file where all the import functions are defined

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
import tifffile as tiff # To read Tiff files
import csv

import multiprocessing
from tqdm import tqdm



###..........................................Functions..........................................###
def delta_intensity(data_file_path, coords_of_interest, ROIrad_for_fit, radius_for_mask):
    
    delta_int = [] # Store the arrays which contain information about the change of intensity in the different spots
    
    for i in range(len(data_file_path)-1):
            
        #Get the data from the files
        file_1 = tiff.imread(data_file_path[i])
        file_2 = tiff.imread(data_file_path[i+1])
        
        shape_of_data = np.shape(file_1) # calculate shape (the shape of file_1 and file_2 are equal)
        
        #Calculate intensity of the coordinates in file_1:
        fitted_coords_1 = func.fit_data(file_1, coords_of_interest, ROIrad_for_fit)
        intensity_1 = func.intensity_in_image(fitted_coords_1, radius_for_mask, shape_of_data, file_1) # Use the intensity function
        
        #Calculate intensity of this coordinate in file_2:
        fitted_coords_2 = func.fit_data(file_2, coords_of_interest, ROIrad_for_fit)
        intensity_2 = func.intensity_in_image(fitted_coords_2, radius_for_mask, shape_of_data, file_2)
        
        if (intensity_1 >=intensity_2):
                average_int = intensity_1-intensity_2
        else: average_int = intensity_2-intensity_1
        delta_int.append([i,average_int])
    
    return np.array(delta_int)


def background_delta_intensity(data_file_path, coords_of_interest, radius_for_mask_main, radius_for_mask_cdw):
    
    delta_int = [] # Store the arrays which contain information about the change of intensity in the different spots
    
    for i in range(len(data_file_path)-1):
            
        #Get the data from the files
        file_1 = tiff.imread(data_file_path[i])
        file_2 = tiff.imread(data_file_path[i+1])
        
        shape_of_data = np.shape(file_1) # calculate shape (the shape of file_1 and file_2 are equal)
        
        #Calculate intensity of the coordinates in file_1:
        intensity_1 = func.background_intensity_in_image(coords_of_interest, radius_for_mask_main, radius_for_mask_cdw, shape_of_data, file_1) # Use the intensity function
        
        ##Calculate intensity of the coordinates in file_1:
        intensity_2 = func.background_intensity_in_image(coords_of_interest, radius_for_mask_main, radius_for_mask_cdw, shape_of_data, file_2) # Use the intensity function
        
        if (intensity_1 >=intensity_2):
                average_int = intensity_1-intensity_2
        else: average_int = intensity_2-intensity_1
        delta_int.append([i,average_int])
    
    return np.array(delta_int)

def create_argument_tuples(file_path, file_chunks, coords_of_interest, ROIrad_for_fit, radius_for_mask ): 
    argument_tuples = []
    for i in range(len(file_chunks)):
        data_info = []
        for name in file_chunks[i]:
            data_info.append(file_path+'\\'+name)
        argument_tuples.append((data_info,coords_of_interest, ROIrad_for_fit, radius_for_mask))
    return argument_tuples

def background_create_argument_tuples(file_path, file_chunks, coords_of_interest, ROIrad_for_fit, radius_for_mask_main, radius_for_mask_cdw ): 
    argument_tuples = []
    for i in range(len(file_chunks)):
        data_info = []
        for name in file_chunks[i]:
            data_info.append(file_path+'\\'+name)
        argument_tuples.append((data_info, coords_of_interest, radius_for_mask_main, radius_for_mask_cdw))
    return argument_tuples

def average_delta_int(delta_int_array):
    return (np.array(delta_int_array).sum())/(len(delta_int_array))

#Check whether the current script is being run as the main program or if it's being imported as a module into another script
if __name__ == '__main__':
    
    ###..........................................Data..........................................###
    drift_corrected_file_path = r'Data\Processed_Data\A13_processed\Drift_corrected_images\RScan4F' # File path where the drift corrected images are stored

    #ATTENTION!!! The integrated images must have the same order as the drift_corrected data
    #Means: For a reprate scan: the right order of the reprate images; For a fluence scan: the right order of the fluence data
    #Convention here is starting with 0kHz(0mW) and going to 2033kHz (xmW) 
    integrated_data = r'Data\Processed_Data\A13_processed\Integrated_images\Reprate_scan_raw_4F'
    integrated_data_files = os.listdir(integrated_data)
    zero_data_name = '0_final_0kHz_0mW_drift_corrected_summed_intensity.tif'

    number = -1 # To iterate through the integrated_data_array
    ROIrad_for_fit = 30 # To fit the data on the coords
    radius_for_mask_main = 38 #Radius for the mask
    radius_for_mask_cdw = 15
    num_cores = 20 # Defines the number of CPU cores which will be in usage in the multiprocessing process
    final_delta_intensity = [] # Array to store all the calculate delta intensity data
    
    ###....................................................................Program....................................................................###
    # Find the spot coordinates of 0mW/0kHz
    zero_khZ_spot_coords = func.find_clustered_diffraction_spots_by_clicking(integrated_data , zero_data_name)
    cdw_unit = zero_khZ_spot_coords[5]
    main_unit = zero_khZ_spot_coords[6]
    
    
    for drift_files in tqdm(os.listdir(drift_corrected_file_path)):
        number+=1 # iterate over number
        data_raw = tiff.imread(integrated_data +'/'+integrated_data_files[number]) #raw data
        
        path = drift_corrected_file_path + "/" + drift_files
        files = os.listdir(path)
        file_chunks = [files[i:i + len(files) // (num_cores-1)] for i in range(0, len(files), len(files) // (num_cores-1))]
        
        #Now find and cluster the diffraction spots
        #spot_coords returns: (1) Beam coordinates, (2) Main spots coordinates, (3) CDW spots coordinates
        #Now find and cluster the diffraction spots
        # Spot_coords returns: (1) Beam coordinates, (2) Main spots coordinates, (3) CDW spots coordinates, (4) CDW  coordinates and false coordinates, (5) All main coordinates
        spot_coords = func.find_clustered_diffraction_spots_by_clicking(integrated_data , integrated_data_files[number])
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
            combined_coords.append(cdw_coords_and_false_spots)
            combined_coords.append([])
            
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
        
        
        # Create a pool of worker processes
        with multiprocessing.Pool(processes=20) as pool:
            # Prepare a list of argument tuples (x, y)
            arguments =  create_argument_tuples(path, file_chunks, main, ROIrad_for_fit, radius_for_mask_main)
            # Use pool.map to apply the function to the argument tuples in parallel
            results = pool.starmap(delta_intensity, arguments)

            combined_result_main = []
            for array in results:
                for element in array:
                    combined_result_main.append(element[1])
                        
            average_delta_int_main = average_delta_int(combined_result_main)
            #max_main_value = np.max(np.array(combined_result_main))
            #min_main_value = np.min(np.array(combined_result_main))
        
        # Calculate the intensity change of the cdw spots
        
        # Create a pool of worker processes
        with multiprocessing.Pool(processes=20) as pool:
            # Prepare a list of argument tuples (x, y)
            arguments =  create_argument_tuples(path, file_chunks, cdw, ROIrad_for_fit, radius_for_mask_cdw)
            # Use pool.map to apply the function to the argument tuples in parallel
            results = pool.starmap(delta_intensity, arguments)

            combined_result_cdw = []
            for array in results:
                for element in array:
                    combined_result_cdw.append(element[1])
                
            average_delta_int_cdw = average_delta_int(combined_result_cdw)
            #max_cdw_value = np.max(np.array(combined_result_cdw))
            #min_cdw_value = np.min(np.array(combined_result_cdw))
        
        
        
        # Calculate the intensity change of the background
        
        # Create a pool of worker processes
        with multiprocessing.Pool(processes=20) as pool:
            # Prepare a list of argument tuples (x, y)
            arguments =  background_create_argument_tuples(path, file_chunks, combined_coords, ROIrad_for_fit, radius_for_mask_main, radius_for_mask_cdw)
            # Use pool.map to apply the function to the argument tuples in parallel
            results = pool.starmap(background_delta_intensity, arguments)

            combined_result_background = []
            for array in results:
                for element in array:
                    combined_result_background.append(element[1])
                
            average_delta_int_background = average_delta_int(combined_result_background)
            
            
            
        
        #combined_delta_intensity = np.concatenate((np.array(combined_result_main), np.array(combined_result_cdw)))  # Calculate the delta intensity of all the spots
        #average_delta_intensity = average_delta_int(combined_delta_intensity)
        
        final_delta_intensity.append([integrated_data_files[number], average_delta_int_main, average_delta_int_cdw,average_delta_int_background])
        

    file_path = r'Data\Processed_Data\A13_processed\Delta_Intensity'
    name_delta_intensity_txt = "Fluence_Scan_254_delta_int.txt"

    # Save the Summed and averaged intensity
    with open(file_path + '/' + name_delta_intensity_txt, "w") as file:
        writer = csv.writer(file, delimiter = ',')
        writer.writerow(["info", "average_delta_main", "average_delta_cdw", "average_delta_background"])
        writer.writerows(final_delta_intensity)


