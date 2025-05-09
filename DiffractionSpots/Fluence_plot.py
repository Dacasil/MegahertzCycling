# Import Libraries
import numpy as np
import tifffile as tiff # To read Tiff files
import matplotlib.pyplot as plt
import os
from scipy import optimize # To make a linear fit

import functions as func #Import the functions.py python file where all the import functions are defined

###.............................Data.............................###
file_name_cdw = "cdw_spot_intensities_Fluence_Scan.txt"
file_name_main = "main_spot_intensities_Fluence_Scan.txt"
file_name_background = "background_intensities_Fluence_Scan.txt"

file_path_A15 = r"Data\Processed_Data\Plot_Fluence_and_Reprate_Scan\A15\Fluence_Scan_modified" + '/' + file_name_cdw 
file_path_A13 = r"Data\Processed_Data\Plot_Fluence_and_Reprate_Scan\A13\Fluence_Scan_modified" + '/' + file_name_cdw 
file_path_A7 = r"Data\Processed_Data\Plot_Fluence_and_Reprate_Scan\A7\Fluence_Scan_modified" + '/' + file_name_cdw 
file_path_A4 = r"Data\Processed_Data\Plot_Fluence_and_Reprate_Scan\A4\Fluence_Scan_modified" + '/' + file_name_cdw 

file_path_array = [file_path_A15, file_path_A13, file_path_A7, file_path_A4]
sample_name = ["A15","A13","A7","A4"]

Energy_A15 = 400 # For calculating the Fluence Energy in [mW]
Energy_A13 = 100
Energy_A7 = 30
Energy_A4 = 5

Energy_Array = [Energy_A15, Energy_A13, Energy_A7, Energy_A4] # Array to calculate the Fluence for the Reprate Array


reprate_fluence_folder = []
for i in range(len(file_path_array)):
    reprate_fluence_folder.append(np.loadtxt(file_path_array[i], float, usecols=2, delimiter = ',',skiprows=1))

print(reprate_fluence_folder[1])
    

array_for_x_axis_reprate_scan = [0,101,254,508,677,1016,2033] # Reprates for the reprate scan in [kHz]
array_for_x_axis_reprate_scan_array =[[0,101,254,508,677,2033],[0,101,254,508,677,1016,2033],[0,101,254,508,677,1016,2033],[0,101,254,508,677,1016,2033]]

array_for_x_axis_fluence_scan_A15 = [0,100,200,300,400] # fluences for the fluence scan in [mW] (E)
array_for_x_axis_fluence_scan_A13 = [0,6,13,32,50,63]
array_for_x_axis_fluence_scan_A7 = [0,2.5,4,9,11.5,12.5,14,16.5]
array_for_x_axis_fluence_scan_A4 = [0,0.619,3,3.5,3.8,4,5]
array_for_x_axis_fluence_scan = [array_for_x_axis_fluence_scan_A15, array_for_x_axis_fluence_scan_A13, array_for_x_axis_fluence_scan_A7, array_for_x_axis_fluence_scan_A4]

Reprate_A15 = array_for_x_axis_reprate_scan[6]
Reprate = array_for_x_axis_reprate_scan[2] # Reprate to calculate the fluence
Reprate_array = [2033,254,254,254]
number = -1

x_axis_name_fluence = r'Fluence in $\left[\frac{mJ}{cm^{2}}\right]$'
x_axis_name_reprate = r"Reprate in [$\mathrm{kHz}$]"
x_axis_name = x_axis_name_fluence

# To save the fluence values for the reprate scan
#fluence_label_name = f" {sample_name[i]} at {np.round(Fluence_for_Reprate[i],1)} mJ/cm^2"
#reprate_label_name = f" {sample_name[i]} at {Reprate_array[i]} kHz"

###.........................Preprocessing.........................###

def calc_fluence(E, Reprate):
    # E in [mw], Reprate in [Hz]
    FWHM = 15*10**(-4) # In [cm]
    fluence = E/(Reprate * (10**3) * np.pi * (FWHM**2))
    return fluence

# Calculate Fluences for the Fluence  Scan
fluence_arr = []
for j in range(len(array_for_x_axis_fluence_scan)):
    
    fluence= []
    for E in array_for_x_axis_fluence_scan[j]:
        fluence.append(calc_fluence(E, Reprate_array[j]))
    fluence = np.array(fluence)
    fluence_arr.append(fluence)


#Fluence_for_Reprate_Scan
Fluence_for_Reprate = []
for i in range(len(Energy_Array)):
    Fluence_for_Reprate.append(calc_fluence(Energy_Array[i], 2033))
Fluence_for_Reprate = np.array(Fluence_for_Reprate)


###..........Plot.............###
# figure object erzeugen
plt.figure(figsize=(10,6))

# Achsenbeschriftungen
plt.xlabel(x_axis_name, fontsize="14")
plt.ylabel(r'$\mathrm{Intensity\ in\ [\%]}$', fontsize="14")


color_array = ['violet','yellow','orange','red']


# Messdaten plotten
for i in range(len(reprate_fluence_folder)):
    plt.errorbar(Fluence_for_Reprate[i], reprate_fluence_folder[i], fmt= "o", linewidth=3, ecolor=color_array[i], capsize=3, label = f" {sample_name[i]} at {Reprate_array[i]} kHz", markerfacecolor=color_array[i], markeredgecolor= color_array[i])
    plt.plot(Fluence_for_Reprate[i], reprate_fluence_folder[i], color_array[i])

# Darstellung der Legende
plt.legend(fancybox=False, loc="best")

# Bild zuschneiden, abspeichern
plt.tight_layout()
plt.savefig(r'Data\Processed_Data\Plot_Fluence_and_Reprate_Scan\Plots' + '/' + "Fluence_Scan_modified.tif")
plt.show()
