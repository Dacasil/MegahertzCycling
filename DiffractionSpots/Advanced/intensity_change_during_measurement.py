import functions as func # Import the functions.py python file where all the import functions are defined

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
import tifffile as tiff # To read Tiff files
import csv

import multiprocessing
from tqdm import tqdm
from PIL import Image



###..........................................Functions..........................................###
def intensity(data_file_path, coords_of_interest, ROIrad_for_fit, radius_for_mask):
    
    intensity = [] # Store the arrays which contain information about the change of intensity in the different spots
    
    for i in range(len(data_file_path)):
        
        #Get the data from the files
        file_1 = tiff.imread(data_file_path[i])
        all_intensity = file_1.sum() # The total intensity in the image
        
        shape_of_data = np.shape(file_1) # calculate shape (the shape of file_1 and file_2 are equal)
        
        #Calculate intensity of the coordinates in file_1:
        #fitted_coords_1 = func.fit_data(file_1, coords_of_interest, ROIrad_for_fit)
        intensity_1 = func.intensity_in_image(coords_of_interest, radius_for_mask, shape_of_data, file_1) # Use the intensity function
        normed_intensity = (intensity_1/all_intensity) *100
        intensity.append([i,normed_intensity])
    
    return np.array(intensity)

def background_intensity(data_file_path, coords_of_interest, ROIrad_for_fit, radius_for_mask_main, radius_for_mask_cdw):
    
    intensity = [] # Store the arrays which contain information about the change of intensity in the different spots
    
    for i in range(len(data_file_path)):
            
        #Get the data from the files
        file_1 = tiff.imread(data_file_path[i])
        all_intensity = file_1.sum() # The total intensity in the image
        
        shape_of_data = np.shape(file_1) # calculate shape (the shape of file_1 and file_2 are equal)
        
        #Calculate intensity of the coordinates in file_1:
        intensity_1 = func.background_intensity_in_image(coords_of_interest, radius_for_mask_main, radius_for_mask_cdw, shape_of_data, file_1)[0] # Use the intensity function
        
        normed_intensity = (intensity_1/all_intensity) *100
        intensity.append([i,normed_intensity])
    
    return np.array(intensity)

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
        argument_tuples.append((data_info, coords_of_interest, ROIrad_for_fit, radius_for_mask_main, radius_for_mask_cdw))
    return argument_tuples

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

    drift_corrected_file_path = r'Data\Processed_Data\A7_processed\Drift_corrected_images\last' # File path where the drift corrected images are stored

    #ATTENTION!!! The integrated images must have the same order as the drift_corrected data
    #Means: For a reprate scan: the right order of the reprate images; For a fluence scan: the right order of the fluence data
    #Convention here is starting with 0kHz(0mW) and going to 2033kHz (xmW) 
    integrated_data = r'Data\Processed_Data\A7_processed\Integrated_images\last'
    zero_integrated_data = r'Data\Processed_Data\A7_processed\Integrated_images\FScan1016raw'
    integrated_data_files = os.listdir(integrated_data)
    zero_data_name = "0_0kHz_0mW_drift_corrected_summed_intensity.tif"
    
    #Folder names to save plots and data
    Reprate_Scan_name = "last"
    data_name = "1016kHz"
    
    # Data where to save the standard deviation data
    file_path = r"Data\Processed_Data\A7_processed\Intensity_in_measurement" + '/' + Reprate_Scan_name + '/' + data_name
    name_std_intensity_txt = "standard_deviation_value_Fluence_Scan_1016.txt"
    name_average_intensity_txt = "average_value_Fluence_Scan_1016.txt"
    name_of_sample = "A7_processed"

    
    number= -1 # To iterate through the integrated_data_array
    ROIrad_for_fit = 40 # To fit the data on the coords
    radius_for_mask_main = 55 #Radius for the mask
    radius_for_mask_cdw = 15
    num_cores = 9 # Defines the number of CPU cores which will be in usage in the multiprocessing process
    final_std_value = []
    main_std = [] # save standard deviation for the bragg spots
    cdw_std = [] # save cdw ...
    background_std = [] # save background ...
    info_array = []
    average_main = []
    average_cdw = []
    average_background = []
    
    ###....................................................................Program....................................................................###

    # Find the spot coordinates of 0mW/0kHz
    zero_khZ_spot_coords = func.find_clustered_diffraction_spots_by_clicking(zero_integrated_data , zero_data_name)
    cdw_zero  = zero_khZ_spot_coords[2] # CDW coordinates of the 0mW/0kHz data
    cdw_coords_and_false_spots_zero = zero_khZ_spot_coords[4]
    main = zero_khZ_spot_coords[1]
    cdw_unit = zero_khZ_spot_coords[5]
    main_unit = zero_khZ_spot_coords[6]
    
    for drift_files in tqdm(os.listdir(drift_corrected_file_path)):
        number+=1
        info_array.append(integrated_data_files[number])
        std_array = []
        path = drift_corrected_file_path + "/" + drift_files # path to the drift corrected data
        files = os.listdir(path)
        print("File name of the drift correction")
        print(files[0])
        
        length_of_files  = len(files)
        list_of_time = list(range(length_of_files))
        file_chunks = [files[i:i + len(files) // (num_cores-1)] for i in range(0, len(files), len(files) // (num_cores-1))]
        
        #Now find and cluster the diffraction spots
        # Spot_coords returns: (1) Beam coordinates, (2) Main spots coordinates, (3) CDW spots coordinates, (4) CDW  coordinates and false coordinates, (5) All main coordinates
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
            cdw_coords_and_false_spots = spot_coords[3]
            combined_coords = [] # Coords to get background intensity
            combined_coords.append(main_coords)
            combined_coords.append(cdw_coords_and_false_spots)

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
            
            # all main coords
            combined_coords = [] # To insert in the background intensity function
            combined_coords.append(main_coords)
            combined_coords.append(cdw_coords_and_false_spots)
            

        
        print("Summed Intensity name")
        print(integrated_data_files[number]) # check which data one is looking at
        
        # Remove text to have a save name for the data
        text = integrated_data_files[number]
        words_to_remove = ["_drift_corrected_summed_intensity.tif"]
        cleaned_text = remove_words(text, words_to_remove)
        
        
        
        ###.............Calculate the intensity change of the main spots.............###
        
        # Create a pool of worker processes
        with multiprocessing.Pool(processes=9) as pool:
            # Prepare a list of argument tuples (x, y)
            arguments =  create_argument_tuples(path, file_chunks, main_coords, ROIrad_for_fit, radius_for_mask_main)
            # Use pool.map to apply the function to the argument tuples in parallel
            results = pool.starmap(intensity, arguments)

            combined_intensity_main = []
            for array in results:
                for element in array:
                    combined_intensity_main.append(element[1])
            
            average_intensity_main = (np.array(combined_intensity_main).sum())/(len(files)) # Average intensity per image
            average_intensity_main = round(average_intensity_main, 3)
            std_dev_main = np.std(combined_intensity_main, ddof=1) # Standard deviation
            main_std.append(std_dev_main)
            average_main.append(average_intensity_main)
            
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
            #plt.title("The intensity of bragg spots")
            
            # Messdaten plotten
            plt.plot([np.min(np.array(list_of_time)), np.max(np.array(list_of_time))], [average_intensity_main, average_intensity_main], color='red', linewidth=1, label = 'average')
            plt.errorbar(list_of_time, combined_intensity_main, fmt= 'rs-', linewidth=1, ecolor="red", capsize=3, label = f"Average brag-spot intensity = {average_intensity_main} \nStandard derivation = {std_dev_main}")
            plt.plot(list_of_time, combined_intensity_main, 'r-')
            
            # Darstellung der Legende
            plt.legend(fancybox=False, loc="best")

            # Bild zuschneiden, abspeichern
            plt.tight_layout()
            plt.savefig(r'Data\Processed_Data' + '/' + name_of_sample + r'\Intensity_in_measurement' +'/'+ Reprate_Scan_name + '/' + data_name + '/bragg_spots_' + cleaned_text)
            plt.show()
        
        
        ###.............Calculate the intensity change of the cdw spots.............###
        
        # Create a pool of worker processes
        with multiprocessing.Pool(processes=9) as pool:
            # Prepare a list of argument tuples (x, y)
            arguments =  create_argument_tuples(path, file_chunks, cdw, ROIrad_for_fit, radius_for_mask_cdw)
            # Use pool.map to apply the function to the argument tuples in parallel
            results = pool.starmap(intensity, arguments)

            combined_intensity_cdw = []
            for array in results:
                for element in array:
                    combined_intensity_cdw.append(element[1])
            
            average_intensity_cdw = (np.array(combined_intensity_cdw).sum())/(len(files)) # Average intensity per image
            average_intensity_cdw = round(average_intensity_cdw, 3) #round to 3 digits after the comma
            std_dev_cdw = np.std(combined_intensity_cdw, ddof=1) # Standard deviation
            cdw_std.append(std_dev_cdw)
            average_cdw.append(average_intensity_cdw)
            
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
            #plt.title("The intensity of cdw spots")
            
            # Messdaten plotten
            plt.plot([np.min(np.array(list_of_time)), np.max(np.array(list_of_time))], [average_intensity_cdw, average_intensity_cdw], color='blue', linewidth=1, label = 'average')
            plt.errorbar(list_of_time, combined_intensity_cdw, fmt= 'bs-', linewidth=1, ecolor="blue", capsize=3, label=f"Average cdw-spot intensity = {average_intensity_cdw} \nStandard derivation = {std_dev_cdw}")
            plt.plot(list_of_time, combined_intensity_cdw, 'b-')
            
            # Darstellung der Legende
            plt.legend(fancybox=False, loc="best")

            # Bild zuschneiden, abspeichern
            plt.tight_layout()
            plt.savefig(r'Data\Processed_Data' + '/' + name_of_sample + r'\Intensity_in_measurement' +'/'+ Reprate_Scan_name + '/' + data_name + '/cdw_spots_' + cleaned_text)
            plt.show()   


        ###.............Work with background data.............###
        
        # Create a pool of worker processes
        with multiprocessing.Pool(processes=9) as pool:
            # Prepare a list of argument tuples (x, y)
            arguments =  background_create_argument_tuples(path, file_chunks, combined_coords, ROIrad_for_fit, radius_for_mask_main, radius_for_mask_cdw)
            # Use pool.map to apply the function to the argument tuples in parallel
            results = pool.starmap(background_intensity, arguments)

            combined_intensity_background= []
            for array in results:
                for element in array:
                    combined_intensity_background.append(element[1])
            
            average_intensity_background = (np.array(combined_intensity_background).sum())/(len(files)) # Average intensity per image
            average_intensity_background = round(average_intensity_background, 3)
            std_dev_background = np.std(combined_intensity_background, ddof=1)
            background_std.append(std_dev_background)
            average_background.append(average_intensity_background)
            
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
            #plt.title("The intensity of background spots")
            
            # Messdaten plotten
            plt.plot([np.min(np.array(list_of_time)), np.max(np.array(list_of_time))], [average_intensity_background,average_intensity_background], color='yellow', linewidth=1, label = 'average')
            plt.errorbar(list_of_time, combined_intensity_background, fmt= 'ys-', linewidth=1, ecolor="yellow", capsize=3, label=f"Average background intensity = {average_intensity_background} \nStandard derivation = {std_dev_background}")
            plt.plot(list_of_time, combined_intensity_background, 'y-')
            
            # Darstellung der Legende
            plt.legend(fancybox=False, loc="best")

            # Bild zuschneiden, abspeichern
            plt.tight_layout()
            plt.savefig(r'Data\Processed_Data' + '/' + name_of_sample + r'\Intensity_in_measurement' +'/'+ Reprate_Scan_name + '/' + data_name + '/background_' + cleaned_text)
            plt.show()

    main_std = np.array(main_std)
    cdw_std = np.array(cdw_std)
    background_std = np.array(background_std)
    info_array = np.array(info_array)
    std_dev = np.column_stack((info_array, main_std, cdw_std, background_std))
    average_int = np.column_stack((info_array, average_main, average_cdw, average_background))



    # Save the std deviation intensity
    with open(file_path + '/' + name_std_intensity_txt, "w") as file:
        writer = csv.writer(file, delimiter = ',')
        writer.writerow(["info", "std_dev_main", "std_dev_cdw", "std_dev_background", "average_main_intensity", "avergae_cdw_intensity", "average_background_intensity"])
        writer.writerows(std_dev)
    # Save the average intensity
    with open(file_path + '/' + name_average_intensity_txt, "w") as file:
        writer = csv.writer(file, delimiter = ',')
        writer.writerow(["info", "average_main_intensity", "avergae_cdw_intensity", "average_background_intensity"])
        writer.writerows(average_int)