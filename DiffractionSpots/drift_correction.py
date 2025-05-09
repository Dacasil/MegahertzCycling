import numpy as np # Basics
import matplotlib.pyplot as plt # "..."

import mrcfile # For image processing
from PIL import Image
import magic
import os # "..."

from IPython.display import clear_output # For the 
from scipy.interpolate import interp1d #Use interpolation on the drift data

import cv2 # For image processing and file access


from skimage.registration import phase_cross_correlation #To calculate the phase cross correlation of two images
from skimage.transform import warp, SimilarityTransform #Shifts an image pixel wise
import itk

from sklearn.preprocessing import normalize # To normalize values

import functions as func # Import the functions module where all the functions are stored

import tifffile as tiff

from tqdm import tqdm



###.................Enter data for processing.................###

#Enter Path Name where measured raw data is stored
path = r'Data\Raw_Data\A4\2_1016kHz\1.8mW'

#Insert names
sample_name  = 'A4' # Name of the sample
reprate_name = '1016kHz'
reprate_folder_name = '2_1016kHz'
fluence_name = '1.8mW'

# Constants
pixel_recon_dim = 20 # image reconstruction pixel size (nm), will be used for two-dimensional histograms, larger value -> faster computation (lower accuracy)
upsample_factor = 100 # for subpixel computation, larger value -> better accuracy (slower computation)


###.............................Program to drift correct and integrate the measured data.............................###
bad_pixel_array = []
path_of_bad_pixel = r'Data\Raw_Data\4_bad_pixel'

for pixel in os.listdir(path_of_bad_pixel):
    if func.mrc_filetype(pixel) == True:  # only work with .mrc files
        im = mrcfile.read(path_of_bad_pixel+'/'+pixel) 
        im = im.astype(np.uint8)
        bad_pixel_array.append(im)

bad_pixel_array = np.array(bad_pixel_array)
summed_bad_pixel = np.sum(bad_pixel_array, axis=0)

threshold = 1
mask = (summed_bad_pixel <= threshold)


image_arrays = [] #Save the numpy arrays of the images
document_names = [] #save their names
i = 0
#Load Images into arrays
for file_names in os.listdir(path):
    if func.mrc_filetype(file_names) == True:  # only work with .mrc files
        im = mrcfile.read(path+'/'+file_names) 
        im = im.astype(np.uint8)
        im = im * mask
        image_arrays.append(im)
        flnms = file_names.replace(".mrc","") # save the names of the .mrc files
        document_names.append(flnms)

#create folder
path = 'Data/Processed_Data/'+sample_name+'_processed_test'+'/Drift_corrected_images/'+reprate_folder_name
folder_name = fluence_name
folder_path = os.path.join(path, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

#Calculate shift between the reference image (here the first image in the array)
#Save the data in the drift corrected Folder

for i in tqdm(range(1, len(image_arrays))):
    shift, error, diffphase = phase_cross_correlation(image_arrays[i], image_arrays[0],upsample_factor=upsample_factor)
    registered_image = warp(image_arrays[i], SimilarityTransform(translation= shift[::-1]),preserve_range=True)
    registered_image = registered_image.astype(np.uint32)
    
    tiff.imwrite('Data/Processed_Data/'+sample_name+'_processed'+'/Drift_corrected_images/'+reprate_folder_name+'/'+fluence_name+'/'+reprate_name+'_'+fluence_name+'_''drift_corrected'+document_names[i]+'.tif', registered_image)

#Get the drift corrected images in a numpy array
im_drift_arrays = []
pth = 'Data/Processed_Data/'+ sample_name + '_processed_test/Drift_corrected_images/' + reprate_folder_name +'/'+fluence_name

#Load Images into arrays
for file_names in os.listdir(pth):
    im = tiff.imread(pth+'/'+file_names) 
    im_drift_arrays.append(im)


# Integrate over all the images
combined_intensity_not_normalized = np.sum(im_drift_arrays, axis=0)
print(type(combined_intensity_not_normalized))

tiff.imwrite('Data/Processed_Data/'+ sample_name +'_processed/Integrated_images/'+ reprate_folder_name + '/' + reprate_name + '_' + fluence_name + '_' +'drift_corrected_summed_intensity'+'.tif', combined_intensity_not_normalized)


