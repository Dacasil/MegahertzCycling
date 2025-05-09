# Import Libraries
import numpy as np
import tifffile as tiff # To read Tiff files
import functions as func #Import the functions.py python file where all the import functions are defined
import matplotlib.pyplot as plt
import os
from scipy import optimize # To make a linear fit


###.............................Data.............................###

file_path = r"Data\Processed_Data\A4_processed\Vibrations\Reprate_Scan"
reprate_fluence_folder = os.listdir(file_path) # Array with the Fluence/reprate Folder data

array_for_x_axis_reprate_scan = [101,254,508,677,1016,2033] # Reprates for the reprate scan in [kHz]
array_for_x_axis_fluence_scan = [0.691,3,3.5,3.8,4,5] # fluences for the fluence scan in [mW]

Reprate = array_for_x_axis_reprate_scan[2]
number = -1

name_of_sample = "A4" # Sample name
name_plot_txt = "A4_Reprate_Scan_SD_for_spots_modified.tif" # Name fo the plot
name_of_txt_file = 'standard_deviation_value_Reprate_Scan.txt' # Name of the txt file were the the SD data is saved

x_axis_name_fluence = "Fluence in [mJ/cm^2]"
x_axis_name_reprate = "Reprate in [kHz]"
x_axis_name = x_axis_name_reprate

###.........................Preprocessing.........................###

def calc_fluence(E, Reprate):
    # E in [mw], Reprate in [Hz]
    FWHM = 15*10**(-4) # In [cm]
    fluence = E/(Reprate * (10**3) * np.pi * (FWHM**2))
    return fluence

fluence = []
for E in array_for_x_axis_fluence_scan:
    fluence.append(calc_fluence(E, Reprate))
fluence = np.array(fluence)

diffraction_spots = []
standard_deviation = []

array_for_x_axis = array_for_x_axis_reprate_scan
print(fluence)
###............................Programm............................###

for rep_flu_folder in os.listdir(file_path):
    
    path = file_path + '/'+ rep_flu_folder + '/' + name_of_txt_file
    
    std_data_spots = np.loadtxt(path, float, usecols=5, delimiter = ',',skiprows=1)
    
    # A list to store the found patterns
    spots = []
    print(std_data_spots)
    print("Absatz")

    # Read the file line by line, starting from line 1
    with open(path, 'r') as file:
        lines = file.readlines()[1:]  # Read all lines starting from line 1
        for line in lines:
            if "(" in line and ")" in line:
                start = line.find("(")
                end = line.find(")")
                pattern = line[start:end + 1]
                spots.append(pattern)
    
    std_average = [] # Array to save std
    spot_names = []
    for i in range(0, len(std_data_spots), 2):
        if i + 1 < len(std_data_spots): # Check if next element exists
            std = (std_data_spots[i] + std_data_spots[i+1])/2
            spot = spots[i] + " and " + spots[i+1]
            std_average.append(std)
            spot_names.append(spot)
        
    diffraction_spots.append(spot_names)
    standard_deviation.append(std_average)

diffraction_spots = diffraction_spots[0]
standard_deviation = np.transpose(np.array(standard_deviation))


###..........Show the plot.............###
# create figure object
fig, axs = plt.subplots(3, 3, figsize=(15,11))

color_array = ['green','blue','violet','pink','yellow','orange','red','brown','black']

number = -1

for i in range(3):
    for j in range(3):
        number +=1
        # Axis labelling
        axs[i,j].set_xlabel(x_axis_name, fontsize="10")
        axs[i,j].set_ylabel("Standard deviation in [%]", fontsize="10")

        # Titel for plot
        #plt.title("The intensity of the " + str(h_k_l[i]) + " spot")

        # Plot measurement data
        axs[i,j].errorbar(array_for_x_axis, standard_deviation[number], fmt= "o", linewidth=3, ecolor=color_array[number], capsize=3, label = diffraction_spots[number], markerfacecolor=color_array[number], markeredgecolor= color_array[number])
        axs[i,j].plot(array_for_x_axis, standard_deviation[number], color_array[number])

        # Presentation of the legend
        axs[i,j].legend(fancybox=False, loc="best")


# Leere Teilplots entfernen
for i in range(number + 1, 9):
    fig.delaxes(axs[i])

# Layout anpassen und den Plot anzeigen
plt.tight_layout()
plt.savefig(r'Data\Processed_Data' + '/' + name_of_sample + '_processed' + '/' +'Vibrations' + '/'+ "Plots"+ '/'+ name_plot_txt)
plt.show()