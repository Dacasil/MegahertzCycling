# Basics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import mrcfile  # work with data

from scipy.ndimage import gaussian_filter  # Implement a gaussian filter
from skimage.feature.peak import peak_local_max  # Find local maxima in the image

from IPython.display import clear_output  # For the progressbar

import itertools  # Find the hexagon pattern in the clusters
import math  # "..."

from itertools import product  # Use for unit_cell_2d function

import cv2  # To read and process data
import os  # To read an Folder and iterate over its files

from skued import autocenter  # Could find the center of an UED image

import tifffile as tiff
from tqdm import tqdm
import csv
from PIL import Image

###.................................................................................Functions.................................................................................###

###..........................Basic Functions..........................###


# Median Filtering
# Median filtering is effective for reducing salt-and-pepper noise in images. It replaces each pixel's value with the median value of its neighborhood
def median_filter(image):
    image = image.astype(np.uint8)
    denoised_image = cv2.medianBlur(image, 3)  # Adjust kernel size as needed

    return denoised_image


# Function which filters the image via a gauss
# The advantage of a filtered image is: 1) Noise reduction (noise could manifest random variations in pixel values); 2) Improve gradient information (smoother image can provide more stable and accurate gradient values);
# 3) improved localization (peaks have more localized maxima)
def gauss_filter_image(
    data_median_filtered, GaussFilterSigma1=10, GaussFilterSigma2=16
):
    # The sigma of the Gaussian filters is specified for the Difference-of-Gaussian filter
    # We filter the image twice, with both sigma
    # Images are here converted to float value to ensure negative numbers during subtraction
    Gauss1FilteredImage = gaussian_filter(
        data_median_filtered, sigma=GaussFilterSigma1
    ).astype(float)
    Gauss2FilteredImage = gaussian_filter(
        data_median_filtered, sigma=GaussFilterSigma2
    ).astype(float)

    # The difference of Gaussian is calculated by subtracting the two images
    Filtered_Image = Gauss1FilteredImage - Gauss2FilteredImage
    return Filtered_Image


###.............................Functions for drift correction.............................###


# Proofs that file type is a mrc
def mrc_filetype(file_path):
    return file_path.lower().endswith(".mrc")


# RGB image into greyscale
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


# ..........................Functions to find and identify the electron beam..........................#


# Center of mass
def center_of_mass(image):
    y_coords, x_coords = np.indices(image.shape)
    total_intensity = np.sum(image)
    center_of_mass_x = np.sum(x_coords * image) / total_intensity
    center_of_mass_y = np.sum(y_coords * image) / total_intensity
    return [center_of_mass_x, center_of_mass_y]


# Find Coordinates of the electron beam (000). Use an Gaus filtered image!
def find_beam_coord(image, spots):

    # calculate the center of mass of the image
    center_of_mass_x = center_of_mass(image)[0]
    center_of_mass_y = center_of_mass(image)[1]

    # Now define a rectangle with the center of mass as midpoint
    # Search in this rectangle for the main spot
    mask = np.zeros(np.shape(image), dtype=bool)
    mask[
        int(center_of_mass_y) - 250 : int(center_of_mass_y) + 250 + 1,
        int(center_of_mass_x) - 250 : int(center_of_mass_x) + 250 + 1,
    ] = True
    image = image * mask  # Apply mask on the image

    # Now find contours in the rectangle. The biggest contour in the near of the Center of Mass of the image
    # belongs with high probability to the main electron beam.
    image = image.astype(np.uint8)
    contours, _ = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )  # Find the counters
    if contours == ():
        beam_coord = [1024, 1024]
        return beam_coord
    else:
        max_contour = max(contours, key=cv2.contourArea)  # Find the maximum counter
        x, y, w, h = cv2.boundingRect(max_contour)
        x_coord = x + w // 2  # x and y coordinate of the spot with the max_contour
        y_coord = y + h // 2

        # Now find beam the coord by comparing the coordinate of the max_contour with the found spots
        beam_coord = None
        min_difference = float("inf")  # Set initial value to positive infinity

        for value in spots:
            difference = np.sqrt((value[0] - x_coord) ** 2 + (value[1] - y_coord) ** 2)
            if difference < min_difference:
                min_difference = difference
                beam_coord = value

        return beam_coord


###..........................classify the found intensity spots (into cdw and main lattice spots)..........................###


# Calculates the distance of the diffraction spots to the beam position
def distance_to_beam(beam_coord, spots):
    distance = []
    for value in spots:
        difference = np.sqrt(
            (value[0] - beam_coord[0]) ** 2 + (value[1] - beam_coord[1]) ** 2
        )
        coordinate_and_distance = [value[0], value[1], difference]
        distance.append(coordinate_and_distance)

    return distance


# Calculates euclidian distance
def calculate_distance(p1, p2):
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5


# calculates angle between to vectors
def calculate_angle(p1, p2, p3):
    v1 = [p2[0] - p1[0], p2[1] - p1[1]]
    v2 = [p3[0] - p2[0], p3[1] - p2[1]]
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude_product = calculate_distance(p1, p2) * calculate_distance(p2, p3)
    return math.degrees(math.acos(dot_product / magnitude_product))


# Find hexagon patterns in the data and returns basis vector of this found hexagon
def find_hexagon_pattern(coordinates, mid_point):
    for i in range(0, len(coordinates)):
        distance_to_mid = calculate_distance(mid_point, coordinates[i])
        for j in range(0, len(coordinates)):
            if j == i:
                continue
            distance = calculate_distance(coordinates[i], coordinates[j])
            if math.isclose(distance, distance_to_mid, rel_tol=0.1) == True:
                basis_vector_1 = np.array(coordinates[i]) - np.array(mid_point)
                basis_vector_2 = np.array(coordinates[j]) - np.array(mid_point)
                return [basis_vector_1, basis_vector_2], [
                    coordinates[i],
                    coordinates[j],
                ]
            else:
                continue


# Creates a lattice out of two basis vectors
# Use it for crystal or diffraction lattices
def unit_cell_2D(a, b, mid_point, Nx, Ny):
    # a and b are the basis vector
    latt_coord_x = []
    latt_coord_y = []
    xpos = mid_point[0]
    ypos = mid_point[1]

    # Plus a and plus b direction
    xpos_all_1 = [
        (xpos + n * a[0] + m * b[0]) for n, m in product(range(Nx), range(Ny))
    ]  # Basis vectors span the array
    ypos_all_1 = [
        (ypos + n * a[1] + m * b[1]) for n, m in product(range(Nx), range(Ny))
    ]
    latt_coord_x.append(xpos_all_1)
    latt_coord_y.append(ypos_all_1)

    # Plus a and minus b direction
    xpos_all_2 = [
        (xpos + n * a[0] - m * b[0]) for n, m in product(range(Nx), range(Ny))
    ]
    ypos_all_2 = [
        (ypos + n * a[1] - m * b[1]) for n, m in product(range(Nx), range(Ny))
    ]
    latt_coord_x.append(xpos_all_2)
    latt_coord_y.append(ypos_all_2)

    # Minus a and plus b direction
    xpos_all_3 = [
        (xpos - n * a[0] + m * b[0]) for n, m in product(range(Nx), range(Ny))
    ]
    ypos_all_3 = [
        (ypos - n * a[1] + m * b[1]) for n, m in product(range(Nx), range(Ny))
    ]
    latt_coord_x.append(xpos_all_3)
    latt_coord_y.append(ypos_all_3)

    # Minus a and minus b direction
    xpos_all_4 = [
        (xpos - n * a[0] - m * b[0]) for n, m in product(range(Nx), range(Ny))
    ]
    ypos_all_4 = [
        (ypos - n * a[1] - m * b[1]) for n, m in product(range(Nx), range(Ny))
    ]
    latt_coord_x.append(xpos_all_4)
    latt_coord_y.append(ypos_all_4)

    latt_coord_x = np.array(latt_coord_x).flatten()
    latt_coord_y = np.array(latt_coord_y).flatten()

    lattice_coord = []
    for i in range(len(latt_coord_x)):
        lattice_coord.append([latt_coord_x[i], latt_coord_y[i]])

    return lattice_coord


###...........Functions to filter: Use it for coordinates in a specific interval...........###


# Only take coordinates in a specific range
def check(value, start, stop):
    if start <= value <= stop:
        return True
    return False


# Filter out coordinates and return only these which lie in a specific range
# One is only interested in cdw or main lattice spots which are lying in the range of the data
def is_within_interval(coord):
    start = 50.0
    stop = 1992
    coords_in_intervall = []

    for i in range(len(coord)):
        if (check(coord[i][0], start, stop) == True) & (
            check(coord[i][1], start, stop) == True
        ):
            coords_in_intervall.append(coord[i])

    coords_in_intervall = np.array(coords_in_intervall)
    coords_in_intervall = np.unique(coords_in_intervall, axis=0)
    return np.array(coords_in_intervall)


# Compare coords for false spots
def compare_false_coord(created_coords, found_spots):

    real_spots_coordinates = []
    min_tolerance = 18  # The calculated distances have to be in this tolerance radius
    test_array = [1, 1]  # array to check if two arrays are the same

    for created_coord in created_coords:
        min_difference = float("inf")  # Set initial value to positive infinity
        real_spot = np.array(
            [float("inf"), float("inf")]
        )  # Define the value of the real spot
        for spot in found_spots:
            difference = np.sqrt(
                (spot[0] - created_coord[0]) ** 2 + (spot[1] - created_coord[1]) ** 2
            )
            if difference < min_difference:
                if abs(difference) < min_tolerance:
                    min_difference = difference
                    real_spot = spot

        diff = np.sqrt(
            (real_spot[0] - test_array[0]) ** 2 + (real_spot[1] - test_array[1]) ** 2
        )
        if diff == float("inf"):
            continue
        else:
            real_spots_coordinates.append(real_spot)

    return np.array(real_spots_coordinates)


# Compare coordinates
def compare_coord(created_coords, found_spots):

    real_spots_coordinates = []
    min_tolerance = 80  # The calculated distances have to be in this tolerance radius
    test_array = [1, 1]  # array to check if two arrays are the same

    for created_coord in created_coords:
        min_difference = float("inf")  # Set initial value to positive infinity
        real_spot = np.array(
            [float("inf"), float("inf")]
        )  # Define the value of the real spot
        for spot in found_spots:
            difference = np.sqrt(
                (spot[0] - created_coord[0]) ** 2 + (spot[1] - created_coord[1]) ** 2
            )
            if difference < min_difference:
                if abs(difference) < min_tolerance:
                    min_difference = difference
                    real_spot = spot

        diff = np.sqrt(
            (real_spot[0] - test_array[0]) ** 2 + (real_spot[1] - test_array[1]) ** 2
        )
        if diff == float("inf"):
            continue
        else:
            real_spots_coordinates.append(real_spot)

    return np.array(real_spots_coordinates)


# Take coordinates lying around the electron beam spot
def only_take_centered_coordinates(beam_coord, spot_coords, number_of_spots):
    centered_coords = []
    distance_array = distance_to_beam(
        beam_coord, spot_coords
    )  # calculate the distance to the electron beam
    sorted_distance = np.array(
        sorted(distance_array, key=lambda row: row[2])
    )  # sort these distances
    # Only take a limited number of diffraction spots around the (000) coordinate
    for i in range(number_of_spots):
        centered_coords.append([sorted_distance[i][0], sorted_distance[i][1]])
    return np.array(centered_coords)


def delete_equal_coords(all_coords, coords_to_delete):
    new_coords = []
    j = -1
    for coord in coords_to_delete:
        for i in range(len(all_coords)):
            if calculate_distance(coord, all_coords[i]) != 0:
                new_coords.append(all_coords[i])
                print("test")
    return np.array(new_coords)


###..............Function to get average spot intensity..............###
# Function takes the spot coordinates and calculate the intensity per spot. Then it sums over these intensities and returns the average intensity of all spots
def average_spot_intensity(coord_list, radius, shape_of_data, data):

    # Define the meshgrid which returns two 2D arrays representing X and Y coordinates of all the points
    Y, X = np.ogrid[: shape_of_data[0], : shape_of_data[1]]
    mask = np.zeros((shape_of_data[0], shape_of_data[1]), dtype=bool)  # Define the mask
    intensity_array = []

    for coord in coord_list:
        mask_new = (X - coord[0]) ** 2 + (
            Y - coord[1]
        ) ** 2 <= radius**2  # mask which only contains the info of one spot
        data_masked = data * mask_new  # Use mask on the data
        intensity = data_masked.sum()  # Get the intensity of the spot
        intensity_array.append(intensity)

    average_intensity = (sum(intensity_array)) / (
        len(coord_list)
    )  # Calculate average spot intensity

    return average_intensity


# Calculates the whole intensity
def intensity_in_image(coord_list, radius, shape_of_data, data):
    # Define the meshgrid which returns two 2D arrays representing X and Y coordinates of all the points
    Y, X = np.ogrid[: shape_of_data[0], : shape_of_data[1]]
    mask = np.zeros((shape_of_data[0], shape_of_data[1]), dtype=bool)

    for coord in coord_list:
        mask1 = (X - coord[0]) ** 2 + (Y - coord[1]) ** 2 <= radius**2
        mask = mask | mask1

    masked_image = data * mask
    intensity = masked_image.sum()
    return intensity


# Calculate the intensity fo the background
def background_intensity_in_image(
    coord_list, main_radius, cdw_radius, shape_of_data, data
):
    # Define the meshgrid which returns two 2D arrays representing X and Y coordinates of all the points
    Y, X = np.ogrid[: shape_of_data[0], : shape_of_data[1]]
    mask_main = np.zeros((shape_of_data[0], shape_of_data[1]), dtype=bool)
    mask_cdw = np.zeros((shape_of_data[0], shape_of_data[1]), dtype=bool)

    # main spots
    for coord in coord_list[0]:
        mask1 = (X - coord[0]) ** 2 + (Y - coord[1]) ** 2 <= main_radius**2
        mask_main = mask_main | mask1
    mask_main = np.logical_not(mask_main)

    if np.shape(coord_list[1])[0] == 0:
        masked_image = data * mask_main
        intensity = masked_image.sum()
        return intensity, mask_cdw, mask_main

    for coord in coord_list[1]:
        mask1 = (X - coord[0]) ** 2 + (Y - coord[1]) ** 2 <= cdw_radius**2
        mask_cdw = mask_cdw | mask1
    mask_cdw = np.logical_not(mask_cdw)

    # Mask the data
    masked_image = data * mask_cdw
    masked_image = masked_image * mask_main
    masked_image = np.array(masked_image)

    """
    white_image = Image.new('L', (shape_of_data[0], shape_of_data[0]), 255)
    white_image = white_image * mask_cdw
    white_image = white_image * mask_main
    plt.imshow(white_image,cmap='gray')
    plt.show()"""

    intensity = masked_image.sum()
    return intensity, mask_cdw, mask_main


###...............................................Main functions...............................................###

###...Find intensity peaks and use "phasor-fitting"...###


# This function returns a list with the localization of the intensity peaks in our data
def localization_intensity_spots(
    imageDataMedianFiltered,
    imageDataRaw,
    GaussFilterSigma1=10,
    GaussFilterSigma2=16,
    MinValueLocalMaxStdMultiplier=0.7,
):
    # Difference-of-Gaussian filter

    # Filter the image with both sigma
    # Float image value to ensure negative numbers during subtraction
    Gauss1FilteredImage = gaussian_filter(
        imageDataMedianFiltered, sigma=GaussFilterSigma1
    ).astype(float)
    Gauss2FilteredImage = gaussian_filter(
        imageDataMedianFiltered, sigma=GaussFilterSigma2
    ).astype(float)

    # Difference of Gaussian
    FilteredImage = Gauss1FilteredImage - Gauss2FilteredImage

    # Local maxima finding

    # Use the peak_local_max function to find the local peaks with a minimum value of MinValueLocalMax
    # Returns array of x,y positions where the local maxima are
    localpeaks = peak_local_max(
        FilteredImage,
        min_distance=40,
        threshold_abs=np.std(FilteredImage) * MinValueLocalMaxStdMultiplier,
    )

    # Filter out local peaks that are on the edge of the image, because extracting the ROI cannot happen there
    # If we shift the ROI (region of interest),the hypothesis that the emitter is approximately in the center of the ROI is wrong

    ROIradius = 20
    # Radius for regions of interest (ROI)
    Radius_of_interest = 40

    # List of indices that should be removed
    markForDeletion = []
    for i in range(0, localpeaks.shape[0]):
        if (
            (localpeaks[i][0] <= (Radius_of_interest + 1))
            or (
                localpeaks[i][0]
                >= imageDataMedianFiltered.shape[0] - (Radius_of_interest + 1)
            )
            or (localpeaks[i][1] <= (Radius_of_interest + 1))
            or (
                localpeaks[i][1]
                >= imageDataMedianFiltered.shape[1] - (Radius_of_interest + 1)
            )
        ):
            markForDeletion = np.append(markForDeletion, i)

    # Now delete these indices from the array
    markForDeletion = np.int_(markForDeletion)
    localpeaks = np.delete(localpeaks, markForDeletion, axis=0)

    # Fit the circles to a centered position
    localization_list = np.zeros((len(localpeaks), 2))
    for l in range(0, len(localpeaks)):
        # Extract the ROI
        ROI = FilteredImage[
            localpeaks[l, 0] - ROIradius : localpeaks[l, 0] + ROIradius + 1,
            localpeaks[l, 1] - ROIradius : localpeaks[l, 1] + ROIradius + 1,
        ]
        # Get locations from the phasor function
        t = phasor_fitting(ROI, ROIradius, [localpeaks[l, 0], localpeaks[l, 1]])
        localization_list[l, :] = t

    return localization_list


###...Function for "phasor fitting"...###


# One performs sub-pixel peak localization in the ROI using a technique that involves the first harmonic of the 2D Discrete Fourier transform
def phasor_fitting(ROI, ROIradius, localpeak):
    # 2D Fourier transform over the complete ROI
    # Resulting ROI_F array holds complex values representing the magnitudes and phases of various spatial frequencies present in the ROI
    ROI_F = np.fft.fft2(ROI)

    # Test
    if (ROI_F[0, 1].real) == 0.0:
        return [localpeak[0], localpeak[1]]

    else:
        # Calculate the phase angle of [0,1] and [1,0] (first harmonic) for the sub-pixel x and y values
        # In general: Subpixel positions of peaks in the signal result in phase shifts of corresponding frequency components
        # See more: "https://doi.org/10.1364/OE.20.012729" for the algorithm and "doi.org/10.1063/1.5005899 (Martens et al., 2017)" for the implementation
        xangle = np.arctan(ROI_F[0, 1].imag / ROI_F[0, 1].real) - np.pi

        # Correct in case it's positive
        if xangle > 0:
            xangle -= 2 * np.pi

        # Calculate the position based on the ROI radius:
        # Use the magnitude of the phase angle to calculate the phase shift (direction independent)
        # Formula calculates how many full cycles (2pi) the adjusted phase angle corresponds to
        # (2*np.pi/(ROIradius*2+1))+0.5 scales the subpixel position based on the number of possible phase shifts in relation to a full cycle of 2π
        PositionX = abs(xangle) / (2 * np.pi / (ROIradius * 2 + 1)) + 0.5

        # Do the same for the Y angle and position
        yangle = np.arctan(ROI_F[1, 0].imag / ROI_F[1, 0].real) - np.pi
        if yangle > 0:
            yangle -= 2 * np.pi

        PositionY = abs(yangle) / (2 * np.pi / (ROIradius * 2 + 1)) + 0.5

        # Get the final localization based on the ROI position
        LocalizationX = localpeak[1] - ROIradius + PositionX
        LocalizationY = localpeak[0] - ROIradius + PositionY

        return [LocalizationX, LocalizationY]  # Returns localization of the peak


###...Fit function which use phasor fitting or any other fitting method if implemented...###


def fit_data(image_data_raw, spots, roi_radius):

    localization_list = np.zeros((len(spots), 2))  # List save fitted spot coordinates
    ROIrad = roi_radius  # Radius of interest
    RawData = image_data_raw  # Image_data_raw
    spots = np.array(spots, dtype="i")

    if len(np.shape(spots)) > 1:
        # Loop over all found localizations
        for l in range(0, len(spots)):
            # Extract the ROI - we know it is centered around the spot position with radius ROIradius
            ROI_new = RawData[
                spots[l, 1] - ROIrad : spots[l, 1] + ROIrad + 1,
                spots[l, 0] - ROIrad : spots[l, 0] + ROIrad + 1,
            ]
            if np.shape(ROI_new)[0] == 0:
                continue
            else:
                # Get locations from the phasor function
                x = phasor_fitting(
                    ROI_new, ROIrad, [spots[l, 1], spots[l, 0]]
                )  # Use phasor fitting (here one could also insert an other fitting function)
                localization_list[l, :] = x
    else:
        # Extract the ROI - we know it is centered around the spot position with radius ROIradius
        ROI_new = RawData[
            spots[1] - ROIrad : spots[1] + ROIrad + 1,
            spots[0] - ROIrad : spots[0] + ROIrad + 1,
        ]
        # Get locations from the phasor function
        x = phasor_fitting(
            ROI_new, ROIrad, [spots[1], spots[0]]
        )  # Use phasor fitting (here one could also insert an other fitting function)
        localization_list = x

    return localization_list


###...Intensity-measure taking background noise into account (consequently not necessary for electron diffraction but maybe for other measurements)...###
def photometry_intensity(ROI):
    # First we create empty signal and background maps with the same shape as
    # the ROI.
    SignalMap = np.zeros(ROI.shape)
    BackgroundMap = np.zeros(ROI.shape)

    # Determine the ROI radius from the data
    ROIrad = 35

    # Now we attribute every pixel in the signal and background maps to be belonging either to signal or background based on the distance to the center
    # For this, we loop over the x and y positions
    for xx in range(0, ROI.shape[0]):
        for yy in range(0, ROI.shape[1]):
            # Now we calculate Pythagoras' distance from this pixel to the center
            distToCenter = np.sqrt(
                (xx - ROI.shape[0] / 2 + 0.5) ** 2 + (yy - ROI.shape[1] / 2 + 0.5) ** 2
            )
            # And we populate either SignalMap or BackgroundMap based on this distance
            if distToCenter <= (ROIrad):  # This is signal for sure
                SignalMap[xx, yy] = 1
            elif distToCenter > (ROIrad - 0.5):  # This is background
                BackgroundMap[xx, yy] = 1

    # First we use the BackgroundMap as a mask for the intensity data, and use
    # that to get a list of Background Intensities.
    BackgroundIntensityList = np.ma.masked_array(ROI, mask=BackgroundMap).flatten()
    # And then we take the 56th percentile (or the value closest to it)
    if len(BackgroundIntensityList) > 0:
        BackgroundIntensity = np.percentile(BackgroundIntensityList, 56)
    else:
        BackgroundIntensity = 0

    # To then assess the intensity, we simply sum the entire ROI in the SignalMap
    # and subtract the BackgroundIntensity for every pixel
    SignalIntensity = sum((ROI * SignalMap).flatten())
    SignalIntensity -= BackgroundIntensity * sum(SignalMap.flatten())

    # And we let the function return the SignalIntensity
    return max(0, SignalIntensity)


###....................................................................Functions to find the intensity peaks and cluster them....................................................................###


# Function to find searched coordinates in an image by clicking on them
def clicked_coord(path, name):
    coords_of_interest = []  # save coords of interest

    # Define a function to display the coordinates of
    # the points clicked on the image
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"({x},{y})")
            coords_of_interest.append((x, y))  # Safe coord data
        return coords_of_interest

    # read the input image
    img = tiff.imread(path + "/" + name)  # Read in the image
    img = img.astype(np.uint8)
    median_filter_img = median_filter(img)
    gauss_img = gauss_filter_image(median_filter_img)

    # create a window
    cv2.namedWindow(
        "Point Coordinates", cv2.WINDOW_NORMAL
    )  # Defines the opened image window
    cv2.resizeWindow("Point Coordinates", 1000, 1000)

    # bind the callback function to window
    cv2.setMouseCallback("Point Coordinates", click_event)
    coords = click_event
    # display the image
    while True:
        cv2.imshow("Point Coordinates", img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    return coords_of_interest


# Find the diffraction spots and cluster them
def find_clustered_diffraction_spots_by_clicking(
    name_path_of_data, name_summed_intensity_file
):
    # Function returns: (1) Beam coordinates, (2) Main spots coordinates, (3) CDW spots coordinates

    # Read in data
    data_raw = tiff.imread(
        name_path_of_data + "/" + name_summed_intensity_file
    )  # raw data
    shape_of_data = np.shape(data_raw)
    data_median_filtered = median_filter(data_raw)  # median_filtered
    # print(name_summed_intensity_file)
    ###...Find intensity peaks in the data...###

    # Find all intensity peaks in the given data by using the "localization_intensity_spots" function
    unclustered_intensity_spots = localization_intensity_spots(
        data_median_filtered, data_raw
    )

    # Use the gauss filter function to get an filtered image of the median filtered data
    FilteredImage = gauss_filter_image(
        data_median_filtered, GaussFilterSigma1=10, GaussFilterSigma2=16
    )

    # Find the main beam coordinates with find_beam_coord function
    beam = find_beam_coord(FilteredImage, unclustered_intensity_spots)

    ###...Cluster diffraction peaks...###

    # Now start to cluster the found diffraction peaks. Categorize them into main lattice peaks and cdw peaks and filter out noise
    # For every diffraction spot find the distance to the main beam spot
    distance = distance_to_beam(
        beam, unclustered_intensity_spots
    )  # Output is [x,y,distance to electron beam]
    distance = np.array(distance)

    # Now sort the distances
    sorted_distance = sorted(distance, key=lambda row: row[2])
    sorted_distance = np.array(
        sorted_distance
    )  # It is sorted from small to large distance

    ###...Find clusters of diffraction spots by their distance from the center point...###

    # Parameters
    cluster_range = 50  # inaccuracy spots could have
    current_cluster = [sorted_distance[0]]
    clusters = []

    # Sort distances into clusters
    for i in range(1, len(sorted_distance)):
        if abs(sorted_distance[i][2] - current_cluster[0][2]) <= cluster_range:
            current_cluster.append(sorted_distance[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [sorted_distance[i]]

    # Filter out the clusters which contain information about the main spot and cdw lattice. In our case these are the first two clusters which contain greater equal 6 elements
    j = 0  # Index to control the number of clusters. Only the first two are needed
    clusters_of_interest = []
    for i in range(len(clusters)):
        if len(clusters[i]) >= 6:
            clusters_of_interest.append(clusters[i])
            j += 1
            if j == 2:
                break
            else:
                continue
    if len(clusters_of_interest) > 1:
        # Only use coordinates from the cluster arrays
        # Because the distance is from small to large the second cluster is the cdw cluster and the third the main lattice cluster
        cluster_main_spots = clusters_of_interest[1]
        cluster_main_spots = np.array(cluster_main_spots)
        cluster_main_spots_distances = cluster_main_spots[:, :2]

        cluster_cdw_spots = clusters_of_interest[0]
        cluster_cdw_spots = np.array(cluster_cdw_spots)
        cluster_cdw_spots_distances = cluster_cdw_spots[:, :2]

        # Calculate Basis vector for main and cdw lattice via the find_hexagon_pattern function
        basis_vector_main = find_hexagon_pattern(cluster_main_spots_distances, beam)[0]
        basis_vector_main = np.array(basis_vector_main)
        basis_vector_cdw = find_hexagon_pattern(cluster_cdw_spots_distances, beam)[0]
        basis_vector_cdw = np.array(basis_vector_cdw)

        # Create from these basis vectors the diffraction spot lattice for cdw and main lattice peaks
        # Basis vectors defining the diffraction grid of the main lattice spots
        a_main = basis_vector_main[0]
        b_main = basis_vector_main[1]
        a_cdw = basis_vector_cdw[0]
        b_cdw = basis_vector_cdw[1]

        # Save the unit vectors of the dcw and main lattice
        main_unit_vectors = [a_main, b_main]
        cdw_unit_vectors = [a_cdw, b_cdw]

        # Number of unit cells to be create in x- and y-direction
        NxMain = 10
        NyMain = 7
        NxCDW = 10
        NyCDW = 10
        # Create the lattice coord via the function unit_cell_2D
        main_latt_coord = unit_cell_2D(
            a_main, b_main, beam, NxMain, NyMain
        )  # Grid of main lattice spots

        cdw_latt_coord_raw = unit_cell_2D(
            a_cdw, b_cdw, beam, NxCDW, NyCDW
        )  # Grid of cdw lattice spots and main lattice spots

        # Currently the cdw lattice coordinates contain also the main lattice points -> so one wants to filter out the main lattice spots
        # Only take cdw spots and filter out the main spots:
        # Do this by comparing the spacial distance between all cdw lattice coordinates and the main lattice coordinates (main_latt_coord)
        # Save the coordinates of the cdw lattice which are the closest to the main lattice coordinates
        # These are the coordinates one wants to filter out
        mainSpots_in_cdw_latt = []
        minimum = float("inf")
        rval = 50
        for i in range(len(cdw_latt_coord_raw)):
            for j in range(len(main_latt_coord)):
                dis = calculate_distance(cdw_latt_coord_raw[i], main_latt_coord[j])
                if dis < minimum:
                    spot = cdw_latt_coord_raw[i]
                    minimum = dis
                else:
                    continue

            if minimum < rval:
                minimum = float("inf")
                mainSpots_in_cdw_latt.append(spot)
            else:
                minimum = float("inf")

        # Now Find the cdw lattice spots by removing the "main lattice spots" (mainSpots_in_cdw_latt) from the cdw spots array
        # Do this with a mask
        cdw_latt_coord = []
        values_to_remove = np.intersect1d(mainSpots_in_cdw_latt, cdw_latt_coord_raw)
        mask = np.isin(cdw_latt_coord_raw, values_to_remove)

        for i in range(len(cdw_latt_coord_raw)):
            if (mask[i][0] & mask[i][1]) == False:
                cdw_latt_coord.append(cdw_latt_coord_raw[i])

        # Make them both numpy arrays
        main_latt_coord = np.array(main_latt_coord)
        cdw_latt_coord = np.array(cdw_latt_coord)

        # Use the is_within_interval function to find spots which lie in a specific range
        new_main_latt_coord = is_within_interval(main_latt_coord)
        new_cdw_latt_coord = is_within_interval(cdw_latt_coord)

        # Compare these coordinates with the unfiltered found spots
        # main_latt_coord_compared= compare_coord(main_latt_coord, unclustered_intensity_spots)
        cdw_latt_coord_compared = compare_coord(
            cdw_latt_coord, unclustered_intensity_spots
        )

        main_coord_compared = compare_coord(
            new_main_latt_coord, unclustered_intensity_spots
        )  # Get the bragg spot coordinates

        new_main_coord = fit_data(data_raw, main_coord_compared, 30)
        new_main_coord = np.unique(new_main_coord, axis=0)

        new_cdw_coord = fit_data(data_raw, cdw_latt_coord_compared, 20)
        new_cdw_coord = np.unique(new_cdw_coord, axis=0)

        # Find the false spots
        all_coords = np.concatenate(
            (np.array(main_coord_compared), np.array(cdw_latt_coord_compared)), axis=0
        )
        set1 = {(x, y) for x, y in all_coords}
        set2 = {(x, y) for x, y in np.array(unclustered_intensity_spots)}
        false_spots = np.array(list(set2 - set1))

        # Get coordinates for the background mask
        new_main_latt_coord = np.concatenate(
            (np.array(new_main_latt_coord), np.array(new_main_coord)), axis=0
        )

        # Get CDW and false spots for the background mask
        if np.shape(false_spots)[0] == 0:
            cdw_spots_and_false_spots = np.concatenate(
                (np.array(new_cdw_latt_coord), new_cdw_coord), axis=0
            )
        else:
            cdw_spots_and_false_spots = np.concatenate(
                (np.array(new_cdw_latt_coord), false_spots), axis=0
            )
            cdw_spots_and_false_spots = np.concatenate(
                (cdw_spots_and_false_spots, new_cdw_coord), axis=0
            )

        """
        for l in range(0,len(cdw_spots_and_false_spots)):
            plt.scatter(cdw_spots_and_false_spots[l][0], cdw_spots_and_false_spots[l][1], facecolors='none', edgecolors='g', s=60)
        plt.imshow(data_raw,cmap='gray')
        plt.show()"""

        # False spots mask
        Y, X = np.ogrid[: shape_of_data[0], : shape_of_data[1]]
        mask = np.zeros((shape_of_data[0], shape_of_data[1]), dtype=bool)

        for coord in false_spots:
            mask1 = (X - coord[0]) ** 2 + (Y - coord[1]) ** 2 <= 10**2
            mask = mask | mask1
        mask_false_spots = np.logical_not(mask)

        # plot to check data
        for l in range(0, len(new_cdw_coord)):
            plt.scatter(
                new_cdw_coord[l][0],
                new_cdw_coord[l][1],
                facecolors="none",
                edgecolors="b",
                s=60,
            )
        for l in range(0, len(new_main_coord)):
            plt.scatter(
                new_main_coord[l][0],
                new_main_coord[l][1],
                facecolors="none",
                edgecolors="r",
                s=60,
            )
        for l in range(0, len(false_spots)):
            plt.scatter(
                false_spots[l][0],
                false_spots[l][1],
                facecolors="none",
                edgecolors="y",
                s=60,
            )
        plt.imshow(data_raw, cmap="gray")
        plt.show()

    else:
        print("press n in the following request")

    # Make a choice about the accuracy of the found spots
    choice = input("Do you want to continue? (y/n): ")

    if choice.lower() == "y":
        return (
            beam,
            new_main_coord,
            new_cdw_coord,
            cdw_spots_and_false_spots,
            new_main_latt_coord,
            cdw_unit_vectors,
            main_unit_vectors,
            mask_false_spots,
        )

    # If it doesn't found the right spots
    elif choice.lower() == "n":
        print("Click Beam")  # choose beam coordinates by clicking
        beam = clicked_coord(name_path_of_data, name_summed_intensity_file)
        beam = np.array(beam[0])
        beam = fit_data(data_raw, beam, 40)

        print("Click two CDW coordinates")  # choose main coordinates by clicking
        cdw = clicked_coord(name_path_of_data, name_summed_intensity_file)

        print("Click two bragg coordinates")  # choose cdw coordinates by clicking
        main = clicked_coord(name_path_of_data, name_summed_intensity_file)
        main = fit_data(data_raw, main, 40)
        a_main = np.array(main[0]) - beam
        b_main = np.array(main[1]) - beam
        main_unit_vectors = [a_main, b_main]

        if np.shape(cdw)[0] == 0:
            # Number of unit cells to be create in x- and y-direction
            NxMain = 10
            NyMain = 7

            # Create the lattice coord via the function unit_cell_2D
            main_latt_coord = unit_cell_2D(
                a_main, b_main, beam, NxMain, NyMain
            )  # Grid of main lattice spots

            # Make them both numpy arrays
            main_latt_coord = np.array(main_latt_coord)

            # Use the is_within_interval function to find spots which lie in a specific range
            new_main_latt_coord = is_within_interval(main_latt_coord)

            # Compare these coordinates with the unfiltered found spots
            main_latt_coord_compared = compare_coord(
                main_latt_coord, unclustered_intensity_spots
            )

            new_main_coord = compare_coord(
                new_main_latt_coord, unclustered_intensity_spots
            )
            coordinates_of_main_spot = np.where(np.all(new_main_coord == beam, axis=1))
            indices = coordinates_of_main_spot[0]
            new_main_coord = np.delete(new_main_coord, indices, axis=0)

            # Find the false spots
            all_coords = np.array(new_main_coord)
            all_coords_compared = compare_false_coord(
                all_coords, unclustered_intensity_spots
            )
            all_coords_compared = np.unique(all_coords_compared, axis=0)
            false_spots = np.array(
                [
                    row
                    for row in unclustered_intensity_spots
                    if not any(np.all(row == row2) for row2 in all_coords_compared)
                ]
            )
            print(len(false_spots))

            # Main coords for background intensity
            new_main_latt_coord = np.concatenate(
                (np.array(new_main_latt_coord), new_main_coord), axis=0
            )

            # False spots mask
            Y, X = np.ogrid[: shape_of_data[0], : shape_of_data[1]]
            mask = np.zeros((shape_of_data[0], shape_of_data[1]), dtype=bool)

            for coord in false_spots:
                mask1 = (X - coord[0]) ** 2 + (Y - coord[1]) ** 2 <= 10**2
                mask = mask | mask1
            mask_false_spots = np.logical_not(mask)

            new_cdw_coord = [[1]]
            cdw_spots_and_false_spots = [[1]]
            cdw_unit_vectors = [[1]]
            main_unit_vectors = [[1]]
            alibi_coords = []

            # plot to check data
            for l in range(0, len(new_main_coord)):
                plt.scatter(
                    new_main_coord[l][0],
                    new_main_coord[l][1],
                    facecolors="none",
                    edgecolors="r",
                    s=60,
                )
            for l in range(0, len(false_spots)):
                plt.scatter(
                    false_spots[l][0],
                    false_spots[l][1],
                    facecolors="none",
                    edgecolors="y",
                    s=60,
                )
            plt.imshow(data_raw, cmap="gray")
            plt.show()

            return (
                beam,
                new_main_coord,
                new_cdw_coord,
                false_spots,
                new_main_latt_coord,
                cdw_unit_vectors,
                main_unit_vectors,
                mask_false_spots,
            )

        else:
            cdw = fit_data(data_raw, cdw, 20)
            a_cdw = np.array(cdw[0]) - beam
            b_cdw = np.array(cdw[1]) - beam
            cdw_unit_vectors = [a_cdw, b_cdw]

            # Number of unit cells to be create in x- and y-direction
            NxMain = 10
            NyMain = 7
            NxCDW = 10
            NyCDW = 10
            # Create the lattice coord via the function unit_cell_2D
            main_latt_coord = unit_cell_2D(
                a_main, b_main, beam, NxMain, NyMain
            )  # Grid of main lattice spots

            cdw_latt_coord_raw = unit_cell_2D(
                a_cdw, b_cdw, beam, NxCDW, NyCDW
            )  # Grid of cdw lattice spots and main lattice spots

            # Currently the cdw lattice coordinates contain also the main lattice points -> so one wants to filter out the main lattice spots
            # Only take cdw spots and filter out the main spots:
            # Do this by comparing the spacial distance between all cdw lattice coordinates and the main lattice coordinates (main_latt_coord)
            # Save the coordinates of the cdw lattice which are the closest to the main lattice coordinates
            # These are the coordinates one wants to filter out
            mainSpots_in_cdw_latt = []
            minimum = float("inf")
            rval = 50
            for i in range(len(cdw_latt_coord_raw)):
                for j in range(len(main_latt_coord)):
                    dis = calculate_distance(cdw_latt_coord_raw[i], main_latt_coord[j])
                    if dis < minimum:
                        spot = cdw_latt_coord_raw[i]
                        minimum = dis
                    else:
                        continue

                if minimum < rval:
                    minimum = float("inf")
                    mainSpots_in_cdw_latt.append(spot)
                else:
                    minimum = float("inf")

            # Now Find the cdw lattice spots by removing the "main lattice spots" (mainSpots_in_cdw_latt) from the cdw spots array
            # Do this with a mask
            cdw_latt_coord = []
            values_to_remove = np.intersect1d(mainSpots_in_cdw_latt, cdw_latt_coord_raw)
            mask = np.isin(cdw_latt_coord_raw, values_to_remove)

            for i in range(len(cdw_latt_coord_raw)):
                if (mask[i][0] & mask[i][1]) == False:
                    cdw_latt_coord.append(cdw_latt_coord_raw[i])

            # Make them both numpy arrays
            main_latt_coord = np.array(main_latt_coord)
            cdw_latt_coord = np.array(cdw_latt_coord)

            # Use the is_within_interval function to find spots which lie in a specific range
            new_main_latt_coord = is_within_interval(
                main_latt_coord
            )  # To get the background
            new_cdw_latt_coord = is_within_interval(cdw_latt_coord)
            new_cdw_latt_coord = new_cdw_latt_coord.astype(np.int64)

            # Compare these coordinates with the unfiltered found spots
            main_latt_coord_compared = compare_coord(
                unclustered_intensity_spots, new_main_latt_coord
            )
            main_latt_coord_compared = np.unique(main_latt_coord_compared, axis=0)
            cdw_latt_coord_compared = compare_coord(
                unclustered_intensity_spots, new_cdw_latt_coord
            )
            cdw_latt_coord_compared = np.unique(cdw_latt_coord_compared, axis=0)

            new_main_coord = fit_data(data_raw, main_latt_coord_compared, 40)
            new_main_coord = np.unique(new_main_coord, axis=0)
            # Filter out the beam
            # coordinates_of_main_spot = np.where(np.all(new_main_coord == beam, axis=1))
            # indices = coordinates_of_main_spot[0]
            # new_main_coord = np.delete(new_main_coord, indices, axis=0)

            new_cdw_coord = fit_data(data_raw, cdw_latt_coord_compared, 20)
            new_cdw_coord = np.unique(new_cdw_coord, axis=0)

            # Find the false spots
            all_coords = np.concatenate(
                (np.array(main_latt_coord_compared), np.array(cdw_latt_coord_compared)),
                axis=0,
            )
            all_coords_compared = compare_false_coord(
                all_coords, unclustered_intensity_spots
            )
            all_coords_compared = np.unique(all_coords_compared, axis=0)

            # Get main spot coordinates for the background mask
            new_main_latt_coord = np.concatenate(
                (np.array(new_main_latt_coord), new_main_coord), axis=0
            )
            """
            for l in range(0,len(all_coords_compared)):
                plt.scatter(all_coords_compared[l][0], all_coords_compared[l][1], facecolors='none', edgecolors='y', s=60)
            plt.imshow(data_raw,cmap='gray')
            plt.show()
            """

            false_spots = np.array(
                [
                    row
                    for row in unclustered_intensity_spots
                    if not any(np.all(row == row2) for row2 in all_coords_compared)
                ]
            )
            print(len(false_spots))
            """
            for l in range(0,len(false_spots)):
                plt.scatter(false_spots[l][0], false_spots[l][1], facecolors='none', edgecolors='y', s=60)
            plt.imshow(data_raw,cmap='gray')
            plt.show()
            """

            # Get CDW and false spots for the background mask
            if np.shape(false_spots)[0] == 0:
                cdw_spots_and_false_spots = np.concatenate(
                    (np.array(new_cdw_latt_coord), new_cdw_coord), axis=0
                )
            else:
                cdw_spots_and_false_spots = np.concatenate(
                    (np.array(new_cdw_latt_coord), false_spots), axis=0
                )
                cdw_spots_and_false_spots = np.concatenate(
                    (cdw_spots_and_false_spots, new_cdw_coord), axis=0
                )

            """
            if np.shape(false_spots)[0] == 0:
                cdw_spots_and_false_spots = np.array(new_cdw_latt_coord)
                print("YES")
            else:
                cdw_spots_and_false_spots = np.concatenate((np.array(new_cdw_latt_coord), false_spots))
            """

            """
            for l in range(0,len(cdw_spots_and_false_spots)):
                plt.scatter(cdw_spots_and_false_spots[l][0], cdw_spots_and_false_spots[l][1], facecolors='none', edgecolors='g', s=60)
            plt.imshow(data_raw,cmap='gray')
            plt.show()"""

            # False spots mask
            Y, X = np.ogrid[: shape_of_data[0], : shape_of_data[1]]
            mask = np.zeros((shape_of_data[0], shape_of_data[1]), dtype=bool)

            for coord in false_spots:
                mask1 = (X - coord[0]) ** 2 + (Y - coord[1]) ** 2 <= 10**2
                mask = mask | mask1
            mask_false_spots = np.logical_not(mask)

            # Plot to check data
            for l in range(0, len(new_cdw_coord)):
                plt.scatter(
                    new_cdw_coord[l][0],
                    new_cdw_coord[l][1],
                    facecolors="none",
                    edgecolors="b",
                    s=60,
                )
            for l in range(0, len(new_main_coord)):
                plt.scatter(
                    new_main_coord[l][0],
                    new_main_coord[l][1],
                    facecolors="none",
                    edgecolors="r",
                    s=60,
                )
            for l in range(0, len(false_spots)):
                plt.scatter(
                    false_spots[l][0],
                    false_spots[l][1],
                    facecolors="none",
                    edgecolors="y",
                    s=60,
                )
            plt.imshow(data_raw, cmap="gray")
            plt.show()

            return (
                beam,
                new_main_coord,
                new_cdw_coord,
                cdw_spots_and_false_spots,
                new_main_latt_coord,
                cdw_unit_vectors,
                main_unit_vectors,
                mask_false_spots,
            )


###.............................................................Function to find the standard deviation between diffraction intensities in one image series.............................................................###


# Calculate the change of Intensity through one measurement
def delta_intensity(
    data_file_path, coords_of_interest, ROIrad_for_fit, radius_for_mask
):

    files = os.listdir(data_file_path)  # Read in files from file path

    delta_int = (
        []
    )  # Store the arrays which contain information about the change of intensity in the different spots

    for i in tqdm(range(1, len(files) - 1)):

        # Get the data from the files
        file_1 = tiff.imread(data_file_path + "/" + files[i])
        file_2 = tiff.imread(data_file_path + "/" + files[i + 1])

        shape_of_data = np.shape(
            file_1
        )  # calculate shape (the shape of file_1 and file_2 are equal)

        # Calculate intensity of the coordinates in file_1:
        fitted_coords_1 = fit_data(file_1, coords_of_interest, ROIrad_for_fit)
        intensity_1 = intensity_in_image(
            fitted_coords_1, radius_for_mask, shape_of_data, file_1
        )  # Use the intensity function

        # Calculate intensity of this coordinate in file_2:
        fitted_coords_2 = fit_data(file_2, coords_of_interest, ROIrad_for_fit)
        intensity_2 = intensity_in_image(
            fitted_coords_2, radius_for_mask, shape_of_data, file_2
        )

        if intensity_1 >= intensity_2:
            average_int = intensity_1 - intensity_2
        else:
            average_int = intensity_2 - intensity_1
        delta_int.append([i, average_int])

    average_delta_int = (np.array(delta_int).sum()) / (len(files) - 1)

    return delta_int, average_delta_int
