# In this file, we make predictions on single image chip using model trained, and merge the predicted chips together into a tiff file

# import libraries
from patchify import patchify, unpatchify
from keras.models import load_model
import tensorflow
import segmentation_models as sm
import gdal, ogr, os, osr
import numpy as np

# read satellite image
image_array, park_array = read_file('city_name')  # function in 0_Data_preparation

# load model
loaded_model = load_model('path', compile = False)  # trained model in 3_Model

# compile model
LR = 0.0001
optim = tensorflow.keras.optimizers.Adam(LR)
loss = sm.losses.binary_focal_dice_loss
metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
loaded_model.compile(optim, loss, metrics=metrics)

# predict on each image chip and merge the predicted array together
# to smooth the process, first crop the image_array to let its width and height divisible by chip_size (256 here)
img_patches = patchify(image_array, (256, 256, image_array.shape[2]), step=256)
predicted_patches = []  # initialise output
for i in range(img_patches.shape[0]):
    for j in range(img_patches.shape[1]):
      single_patch_img = img_patches[i, j, 0, :, :, :]
      single_patch_img = normalize_by_layer(single_patch_img)  # function in 0_Data_preparation
      single_patch_img = np.expand_dims(single_patch_img, axis = 0)  # expand dimension to fit the shape of training data of loaded_model
      pred = loaded_model.predict(single_patch_img)  # make prediction on single patch
      pred_argmax = np.argmax(pred, axis = 3)  # get the max value in axis = 3, need to do this because we are using one-hot encoding
      pred_argmax = np.expand_dims(pred_argmax, axis = 3)[0, :, :, :]  # expand dimension to fit shape
      predicted_patches.append(pred_argmax)

# turn list into array
predicted_patches = np.array(predicted_patches)

# reshape array for unpatchify
predicted_patches_reshaped = predicted_patches.reshape((img_patches.shape[0], img_patches.shape[1], 256, 256, 1))
predicted_patches_reshaped = predicted_patches_reshaped[:, :, :, :, 0]  # to fit shape for unpatchify

# merge chips together (unpatchify)
reconstructed_image = unpatchify(predicted_patches_reshaped, (256*img_patches.shape[0], 256*img_patches.shape[1]))

# save predicted array into tiff file
# function to turn array into tiff file with metadata
def array2raster(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array):
    cols = array.shape[1]
    rows = array.shape[0]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Byte)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)  # set epsg you want
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

def main(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array):
    array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array) # convert array to raster

# get metadata of original image
image = gdal.Open('file_path')
geo_transform = image.GetGeoTransform()
originX = geo_transform[0]
originY = geo_transform[3]
pixelWidth = geo_transform[1]
pixelHeight = geo_transform[5]

# save predicted array as tiff file
if __name__ == "__main__":
    rasterOrigin = (originX, originY)
    pixelWidth = pixelWidth
    pixelHeight = pixelHeight
    newRasterfn = 'test.tif'  # file path you want to save
    main(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, reconstructed_image)
