# Park-NET: identifying Urban parks using multi source spatial data and Geo-AI
The main workflow of this project is, 

* First, data preparation, including importing satellite images and corresponding masks (the ground truth for where are parks and where are not), creating chips and balancing data.

* Second, data augmentation. This was done to expand image dataset and reduce overfitting.

* Third, model training and evaluation. In this step, different models and parameters (e.g., image size, combinations of bands) were tested (both on validation set and unseen dataset).

## 1_Data_preparation
Import satellite images have totally nine bands and indices, which are R, G, B, NIR, NDVI, NDBI, NDWI, land surface temperature, land cover. Create chips for satellite images and masks, remove chips with high proportion of backgrounds (over 80%) to get balanced training data. Example is shown as

<img src="https://user-images.githubusercontent.com/97944674/170721870-a8c1c3a8-2df7-417d-890e-b0bf75c3c1ba.png" width="530" height="220">

## 2_Data_augmentation
The python library used was ImageDataGenerator, with augmentation techniques of rotation, shift, flip, etc., with fill model of "reflect". Example is shown as

<img src="https://user-images.githubusercontent.com/97944674/170722580-d01421e3-9c0f-415c-a8cd-22db08122ece.png" width="550" height="230">

## 3_Model_UNet
When training the model on San Francisco, Seattle, Philadelphia, the overall accuracy of this model is 0.93, IoU is 0.75, F1 score is 0.732. When tesing on a new city -- Denver, the overall accuracy is 0.78, IoU is 0.41, F1 score is 0.33.

## 3_Model_ResUNet
Compared with U-Net model from scrath, a pretrained backbone can increase model performace and make the training converge faster. A ResNet50 backbone is used as the encoder part of U-Net model. As the backbone only accepts three-channel images (otherwise you cannot make use of the pretrained weights or you need to reshape your images into 3-channel), three bands were chosen. Here is the example for bands NDVI, NDWI, and land cover

<img src="https://user-images.githubusercontent.com/97944674/170857546-eed7d2bb-2d4c-47bd-a908-abca6a79cc84.png" width="300" height="200"><img src="https://user-images.githubusercontent.com/97944674/170857566-e5a90ee2-dffa-411e-9073-d5add2bbae99.png" width="300" height="200"><img src="https://user-images.githubusercontent.com/97944674/170857586-d774259a-f6a6-4094-a7df-aea94c2e043d.png" width="300" height="200">

When training on San Francisco, Denver, Seattle, Ghent, Greater Manchester, Dhaka, Dublin, and Amsterdam, the model has a overall accuracy of 0.94, IoU of 0.82, F1 score of 0.90. When testing the model in an external city -- Philadelphia, the overall accuracy is 0.90, IoU of 0.63, F1 socre of 0.74.

<img src="https://user-images.githubusercontent.com/97944674/170857470-a8f046b6-3ffe-4dfe-b1db-c7b4c1deced3.png" width="600" height="260">


## 3_Model_BigEarthNet
(IN PROGRESS)

## 4_Prediction_and_save_as_tiff
To make predictions on external satellite images and save the prediction to a tiff file to visualize in GIS applications. In this file, predictions were made for each image chip, and then predicted chips were merged together to a numpy array. This numpy array then was converted into tiff file using given metadata.

## 5_Multi_city_solution_with_GEE
If you are training model on multiple cities, you can use Google Earth Engine to automatically download satellite images, automatically crop satellite images into chips, and create corresponding chips for masks.

Reference: 

https://segmentation-models.readthedocs.io/en/latest/api.html#unet

https://github.com/bnsreenu/python_for_microscopists

https://geemap.org/notebooks/96_image_chips/
