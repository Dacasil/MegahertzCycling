# Import Libraries
import numpy as np
import cv2
import tifffile as tiff # To read Tiff files
import functions as func #Import the functions.py python file where all the import functions are defined
import matplotlib.pyplot as plt
import os
from scipy import optimize # To make a linear fit
import csv

###...................Functions...................###

# Calculates the lattice distance for a Monoclinic system
def d_hkl(h,k,l,beta,a,b,c):
    return 1/(np.sqrt((k**2)/(b**2) + (h**2)/((a**2)*np.sin(beta)**2) + (l**2)/((c**2)*np.sin(beta)**2) - (2*h*l*np.cos(beta))/(a*c*np.sin(beta)**2))) # Distance for the (hkl) lattice planes for a monoclinic unit cell

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

#Definieren der Fit-Funktion
def fit_func(x, m, b):
    return m* x + b


###...................Data................###
path = r'Data\Processed_Data\A15_processed\Integrated_images\1_2033kHz' # Path where the fluence or reprate scan data is stored
zero_mW_name = r'1_2033kHz_0mW_drift_corrected_summed_intensity.tif'

save_data_path = r'Data\Processed_Data\A15_processed\Mean_square_lattice_displacement\Fluence_Scan_2033kHz'
save_file_name = '2033kHz_fluence_mean_square_lattice_displacement.txt'
###...............Constants...............###
h_k_l = [(2,0,2), (1,3,1), (0,6,0), (3,3,3), (4,0,4), (2,6,2)] # absolut values of the h,k,l values
alpha, beta, gamma = [90, 110.926, 90] # values of the lattice angles
a,b,c = [14.784, 3.637, 9.345] # lattice constants in Angstrom
I_zero = [] # array to save the intensity of the unexcited data (here zero_mW_name)
u_square = [] # Mean square displacement of an atom
data_files = os.listdir(path)

###.....................................Program.....................................###
for i in range(len(h_k_l)):
    
    hkl = h_k_l[i] # Diffraction spot of interest
    print(hkl) # print the spot to know which one to click on
    
    zero_data = tiff.imread(path +'/'+ zero_mW_name) # Read the 0mW/0kHz data

    #Find coords of interest by clicking on the image
    zero_coords_of_interest = clicked_coord(path, zero_mW_name)

    #Find diffraction spots
    diffraction_spots_zero = func.find_clustered_diffraction_spots(path, zero_mW_name)
    main_spots_zero = diffraction_spots_zero[1]
    beam_zero = diffraction_spots_zero[0]

    #Compare the coordinates
    zero_coords_of_interest = np.array(zero_coords_of_interest)
    zero_specific_spot_coord = func.compare_coord(zero_coords_of_interest, main_spots_zero)

    radius = 65 # radius for intensity
    shape_of_data = np.shape(zero_data) # shape of the data
    I_0 = func.intensity_in_image(zero_specific_spot_coord, radius, shape_of_data, zero_data)
    I_zero.append(I_0)


for j in range(1, len(data_files)):
    
    debye_waller = [] # array to store the debye waller data
    x_axis = [] # x-axis data
    y_axis = [] # y-axis data
    y_error = [] # error of y_axis value
    delta_I = 6800
    
    data = tiff.imread(path +'/'+ data_files[j]) # read in the excited image data
    print(data_files[j])
    
    for i in range(len(h_k_l)):
        hkl = h_k_l[i] # the spot
        dhkl = d_hkl(hkl[0], hkl[1], hkl[2],beta,a,b,c) # Calculate the lattice distance
        
        print(hkl) # print the spot to know which one to click on
        
        #Find coords of interest by clicking on the image
        fluence_coords_of_interest = clicked_coord(path, data_files[j])
        if fluence_coords_of_interest == []:
            continue
        #Find diffraction spots
        diffraction_spots = func.find_clustered_diffraction_spots_by_clicking(path, data_files[j])
        main_spots_fluence = diffraction_spots[1]
        beam_fluence = diffraction_spots[0]

        #Compare the coordinates
        fluence_coords_of_interest = np.array(fluence_coords_of_interest)
        fluence_specific_spot_coord = func.compare_coord(fluence_coords_of_interest, main_spots_fluence)
        print(fluence_specific_spot_coord)
        radius = 65 # radius for intensity
        shape_of_data = np.shape(data) # shape of the data
        I_Y = func.intensity_in_image(fluence_coords_of_interest, radius, shape_of_data, data)

        #Save the data in the debye_waller array
        array = [(hkl[0], hkl[1], hkl[2]), dhkl, I_zero[i], I_Y] # array containing 1) hkl values 2) d_hkl lattice distance 3) I_0 intensity 4) I_Y intensity
        debye_waller.append(array)
    
    # x-axis
    for i in range(len(debye_waller)):
        x_axis.append((1/((debye_waller[i][1])**2)) * (4*np.pi**2)/3) # 4pi^2/3 * 1/d_hkl^2
    # y-axis
    for i in range(len(debye_waller)):
        y_axis.append(-np.log(debye_waller[i][3]/debye_waller[i][2])) # -log(I_y/I_0)
        print("y_value")
        print(debye_waller[i][2])
        print(debye_waller[i][3])
    # Calculate the error
    for i in range(len(debye_waller)):
        y_error.append(np.square((delta_I/debye_waller[i][3])**2 + (delta_I/debye_waller[i][2])**2))
        print("Y-Error")
        print(np.square((delta_I/debye_waller[i][3])**2 + (delta_I/debye_waller[i][2])**2))
        
    
    params_name = ["m", "b"]
    # Aufruf der Fittenden-Funktion von optimize
    params, params_covariance = optimize.curve_fit(fit_func, x_axis, y_axis, p0=[0,0])
    
    # Die Ergebnisse des Fits printen
    for i in range(0, len(params)):
        print(str(params[i]) + " +- " + str(np.sqrt(params_covariance[i][i])))

    
    u_square.append([data_files[j], params[0], params_covariance[0][0]]) # save the mean square displacement of an atom due to laser excitation. here it is the slope of the fit line
    
    ###..........Show the plot.............###
    # figure object erzeugen
    plt.figure(figsize=(10,6))

    #Gridlines erzeugen (wie man sieht, gibt es zig Möglichkeiten diese zu Zeichnen)
    plt.minorticks_on()
    plt.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.6)
    plt.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.3)

    # Achsenbeschriftungen
    plt.xlabel("1/d_hkl^2", fontsize="14")
    plt.ylabel("I_Y/I_0", fontsize="14")

    # Titel für den Plot
    plt.title("Debye-Waller")

    xlo = min(x_axis)
    xhi = max(x_axis)
    xplot = np.linspace(xlo, xhi, endpoint=True)

    # Messdaten plotten
    plt.errorbar(x_axis, y_axis, y_error, fmt= '.r', linewidth=1, ecolor="red", capsize=3)
    k = -1
    for (i, j) in zip(x_axis, y_axis):
        k+=1
        plt.text(i, j, f'({h_k_l[k]})')

    # Fit-Kurve plotten
    plt.plot(xplot, fit_func(xplot, *params), color='red', linewidth=1)

    # Darstellung der Legende
    plt.legend(fancybox=False, loc="best")

    # Bild zuschneiden, abspeichern
    plt.tight_layout()
    plt.show()

with open(save_data_path + '/' + save_file_name , "w") as file:
    writer = csv.writer(file, delimiter = ',')
    writer.writerow(["Info", "Mean square displacement of an atom", "Fehler"])
    writer.writerows(u_square)