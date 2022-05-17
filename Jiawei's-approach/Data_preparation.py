import os
import rasterio
import numpy as np
from osgeo import gdal


def read_file(dir, city, image_file_name, park_file_name):
    '''
    read the satellite image file and corresponding mask as np.array of a given city
    '''

    # read image file
    image_file_path = os.path.join(dir, city, image_file_name)
    image = gdal.Open(image_file_path)
    image_array = image.ReadAsArray()
    image_array = np.transpose(image_array, [1, 2, 0])  # transpose the first and third axis
    image_array[np.isnan(image_array)] = 0  # replace nan with 0

    # read park file
    park_file_path = os.path.join(dir, city, park_file_name)
    park = gdal.Open(park_file_path)
    park_array = park.ReadAsArray()
    park_array = np.expand_dims(park_array, axis=2)  # expand from 2D to 3D
    park_array[np.isnan(park_array)] = 0  # replace nan with 0
    park_array[park_array > 1] = 1  # repalce values greater than 1 with 1

    return image_array, park_array


from PIL import Image
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# define a function to crop images and corresponding masks into proper size
def create_chips(image_file, park_file, patch_size, step):
    '''
    This function creates chips for satellite image and corresponding masks
    Input  - image_file: np.array of satellite image
           - park_file: np.array of mask
           - patch_size: size of output chips
           - step: stride when cropping
    Output - one np.array for chips of satellite image, another one for mask
    '''

    scaler = MinMaxScaler()  # scale to the min = 0, and max = 1

    # crop image_file
    image_dataset = []
    patches_img = patchify(image_file, (patch_size, patch_size, image_file.shape[2]), step=step)

    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i, j, :, :]
            # scale to min = 0, max = 1
            single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(
                single_patch_img.shape)
            single_patch_img = single_patch_img[0]  # Drop the extra unecessary dimension that patchify adds.
            image_dataset.append(single_patch_img)
    image_dataset = np.array(image_dataset)

    # cropping park_file
    park_dataset = []
    patches_img = patchify(park_file, (patch_size, patch_size, park_file.shape[2]), step=step)

    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i, j, :, :]
            # scale to min = 0, max = 1
            single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(
                single_patch_img.shape)
            single_patch_img = single_patch_img[0]  # Drop the extra unnecessary dimension that patchify adds.
            park_dataset.append(single_patch_img)
    park_dataset = np.array(park_dataset)

    return image_dataset, park_dataset


def remove_images(image_dataset, park_dataset, threshold):
    '''
    This function remove images and corresponding masks with high proportion of backgrounds
    Input  - image_dataset, park_dataset: np.array of chips of satellite images and masks
           - threshold: if proportion of backgrounds is higher than threshold, the chips will be removed
    Output - balanced array without those chips
    '''

    # get the id of images to remove
    id_to_remove = []
    for i in range(len(park_dataset)):
        mask = park_dataset[i, :, :, 0]
        tot_pixel = mask.size
        background_pixel = np.count_nonzero(mask == 0)
        if background_pixel > tot_pixel * threshold:
            id_to_remove.append(i)

    # get the balanced dataset
    image_dataset_balanced = []
    park_dataset_balanced = []
    for i in range(len(image_dataset)):
        if not (i in id_to_remove):
            image = image_dataset[i]
            park = park_dataset[i]
            image_dataset_balanced.append(image)
            park_dataset_balanced.append(park)
    image_dataset_balanced = np.array(image_dataset_balanced)
    park_dataset_balanced = np.array(park_dataset_balanced)

    return image_dataset_balanced, park_dataset_balanced
