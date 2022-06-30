# Automatically Mapping Urban Green Space Using Sentinel-2 Imagery and Deep Learning Methods in Multiple Cities Worldwide: A Convolutional Neural Network Approach

The main workflow of this project is, 

* First, data preparation, including importing satellite images and corresponding masks (the ground truth for where are urban green spaces (UGSs) and where are not), creating chips, balancing data, and data augmentation.

* Second, model building and training. This was done on the 13 chosen cities from different continents: San Francisco, Seattle, Denver, Philadelphia, Greater Manchester, Dublin, Amsterdam, Ghent, Dhaka, Vancouver, Dallas, London, and Buffalo. Both U-Net model from base level and with pretrained backbones were tested, with multiple combinations of input bands.

* Third, model validation and prediction on external cities. In this step, models with the best performance would be used to validate on Washington D.C. and Tel Aviv, and predict on Kampala.

## Data preparation
Satellite images were generated from Sentinel-2, downloaded at [EarthExplorer](https://earthexplorer.usgs.gov/), with spectral information of blue (B2), green (B3), red (B4), near-infrared (NIR, B8). Together with these bands, [NDVI](https://en.wikipedia.org/wiki/Normalized_difference_vegetation_index), [NDBI](https://pro.arcgis.com/en/pro-app/2.8/arcpy/spatial-analyst/ndbi.htm), [NDWI](https://en.wikipedia.org/wiki/Normalized_difference_water_index) were also calculated and put into training. Additionally, we got landcover data and added this as another layer.

Functions defined in this step include `read_file()`, `normalize_by_layer()`, `create_chips()`, `remove_images()`, with detailed explanation shown in the table below. Python scripts are in [1_Data_preparation](https://github.com/mar-koz22/Park-NET-identifying-Urban-parks-using-multi-source-spatial-data-and-Geo-AI/blob/main/Jiawei's-approach/1_Data_preparation.py). 

| Function | Description | Inputs | Parameters set in this study |
| -------- | ------ | ------- | ---------------------------- |
|*read_file()*|read the satellite image file and corresponding mask as np.array of a given city | *dir, city, image_file_name, park_file_name*: directory and file name of each image| |
|*normalize_by_layer()*|to normalize image data to the same max and min. Since different layers have different scales, normalization will be done layer by layer|*image_array*: np.array of the image| set *max* to be 1 and *min* to be 0
|*create_chips()*|create chips for satellite image and corresponding masks|*image_file, park_file*: np.array for satellite image and mask <br />*patch_size*: the number of pixels of each chip <br /> *step*: stride, which represents the distance travelled during the motion|*patch_size* = 256 pixels <br /> *step* = 32, 64, 80 pixles for different cities to make each training city has similar contribution|
|*remove_images()*|remove images and corresponding masks with high proportion of backgrounds (non-UGS pixels) to balance data|*image_dataset, park_dataset*: np.array of image and mask <br /> *threshold*: chip pairs with the proportion of backgrounds higher than the threshold are removed|*threshold* = 90% for Dhaka and 86% for other cities chosen from the trade-off between the training set size and degree of balance

Examples of image chip pairs are: 

<img src="https://user-images.githubusercontent.com/97944674/174822286-3e415b46-2be7-4ee5-8650-494150aaaa2f.png" width="650" height="325">

Techniques chosen for data augmentation are (1) random rotation within an angle of 45 degrees; (2) random width and height shift within a range of 20%; (3) randomly horizontal and vertical flip; (4) randomly zooming in and out within a range of 20%. The python library used was `ImageDataGenerator`. Python scripts are in [2_Data_augmentation](https://github.com/mar-koz22/Park-NET-identifying-Urban-parks-using-multi-source-spatial-data-and-Geo-AI/blob/main/Jiawei's-approach/2_Data_augmentation.py). Example is shown as

<img src="https://user-images.githubusercontent.com/97944674/176749354-3e2c1aaf-7a0c-42e0-9092-94b68ab7e015.png" width="650" height="325">

## Model training
### U-Net model from scratch
U-Net model from scratch was built in [3_Model_UNet](https://github.com/mar-koz22/Park-NET-identifying-Urban-parks-using-multi-source-spatial-data-and-Geo-AI/blob/main/Jiawei's-approach/3_Model_UNet.py), with the number of filters chosen to be (64, 128, 256, 512, 1024). Due to limited system memory, eight three-bands combinations were used instead of all the 8 layers. Model performance from different combinations are:

| Combination of bands | OA     | IoU    | F-score | AUC    |
| -------------------- | ------ | ------ | ------- | ------ |
| Red-Green-Blue       | 0.8329 | 0.5394 | 0.5438  | 0.6894 |
| Red-Green-NIR        | 0.8531 | 0.6425 | 0.6227  | 0.7373 |
| NDVI-Red-NIR         | 0.8724 | 0.6429 | 0.6921  | 0.7745 |
| NDWI-Red-NIR         | 0.8977 | 0.7185 | 0.7819  | 0.8457 |
| NDBI-Red-NIR         | 0.8498 | 0.6072 | 0.6583  | 0.7599 |
| NDVI-NDWI-NDBI       | 0.8515 | 0.5846 | 0.6126  | 0.7252 |
| NDVI-NDWI-landcover  | 0.8502 | 0.5964 | 0.6365  | 0.7431 |
| NDVI-NDBI-landcover  | 0.8914 | 0.7044 | 0.7682  | 0.8368 |
| Average              | 0.8624 | 0.6295 | 0.6645  | 0.7640 |

### U-Net model with pretrained backbones
Compared with U-Net model from scrath, a pretrained backbone can increase model performace and make the training converge faster. ResNet-50 and VGG-16 pretrained on ImageNet were used as the encoder part of U-Net model in [3_Model_UNet_with_backbones](https://github.com/mar-koz22/Park-NET-identifying-Urban-parks-using-multi-source-spatial-data-and-Geo-AI/blob/main/Jiawei's-approach/3_Model_UNet_with_backbones.py), with performance on training cities shown as:

| Combination of bands |ResNet-50 |        |         |        | VGG-16		|        |         |         |
| -------------------- | ---------| ------ | ------- | ------ | ----------| ------ | ------- | ------- |
|                      | OA       | IoU    | F-score | AUC    | OA        | IoU    | F-score | AUC	   |
| Red-Green-Blue       | 0.9708   | 0.9220 | 0.9590  | 0.9610 | 0.2535    | 0.1243 | 0.2001  | 0.5013	 |
| Red-Green-NIR        | 0.9718   | 0.9243 | 0.9602  | 0.9596 | 0.7508    | 0.3769 | 0.4319  | 0.4996	 |
| NDVI-Red-NIR         | 0.9709   | 0.9223 | 0.9591  | 0.9608 | 0.7259    | 0.3857 | 0.4588  | 0.4952	 |
| NDWI-Red-NIR         | 0.9712   | 0.9225 | 0.9592  | 0.9616 | 0.2804    | 0.1456 | 0.2409  | 0.5063	 |
| NDBI-Red-NIR         | 0.9704   | 0.9208 | 0.9583  | 0.9607 | 0.7439    | 0.3766 | 0.4348  | 0.4980  |
| NDVI-NDWI-NDBI       | 0.9691   | 0.9178 | 0.9566  | 0.9565 | 0.2476    | 0.1222 | 0.1960  | 0.5000	 |
| NDVI-NDWI-landcover  | 0.9660   | 0.9107 | 0.9526  | 0.9532 | 0.6020    | 0.3581 | 0.4834  | 0.4768	 |
| NDVI-NDBI-landcover  | 0.9656   | 0.9100 | 0.9522  | 0.9536 | 0.2468    | 0.1222 | 0.1961  | 0.5000	 |
| Average              | 0.9695   | 0.9188 | 0.9572  | 0.9584 | 0.5139    | 0.2515 | 0.3303  | 0.4972	 |

ResNet-50 backbone added much benefit to the model, while VGG-16 made the model perform worse. The converge histories of the three models are plotted below. ResNet-50 helps the training converge faster, however VGG-16 has difficulty to converge within the 50 epochs.

<img src="https://user-images.githubusercontent.com/97944674/176749834-8c2c5f73-422d-40b2-b14c-0469a241a424.png" width="700" height="370">

## Validation & prediction on external cities
U-Net with pretrained ResNet-50 backbone was used to do validation. The average OA, IoU, F-score, and AUC for Washington D.C. are 0.8743, 0.6185, 0.7236, and 	0.7429, for Tel Aviv are 0.8790, 0.4954, 0.5660, and 0.5639, showing a moderate to good generalization capacity of our model.

<img src="https://user-images.githubusercontent.com/97944674/176751670-56bca353-13f8-44da-850d-925696ecff86.png" width="700" height="620">

We also used Washington D.C. as an example to explore where our model performed well and where could be improved, with results shown below. We can see that the model made a good prediction for clumped UGSs, which are circled in red. However, for UGSs which have buildings inside, for instance, the blue areas in rectangle, our model did not give an accurate identification. Additionally, the model also generated some UGSs which are not in the ground truth dataset, mainly golf courses, national parks, and cemeteries, which are not freely accessible to the public or not normal UGSs under the hard definition of UGS.

<img src="https://user-images.githubusercontent.com/97944674/176751827-33f024a4-4817-4091-81f9-d87282595fde.png" width="700" height="370">

Python scripts to do this are documented in [4_Prediction_and_save_as_tiff](https://github.com/mar-koz22/Park-NET-identifying-Urban-parks-using-multi-source-spatial-data-and-Geo-AI/blob/main/Jiawei's-approach/4_Prediction_and_save_as_tiff.py). Predictions were made for each image chip, and then predicted chips were merged together to a numpy array. This numpy array then was converted into tiff file using given metadata. Main functions used were `unpatchify` in `patchify` library and the defined `array2raster`.

## Multi cities solution
If you are training model on multiple cities, you can use Google Earth Engine to automatically download satellite images, automatically crop satellite images into chips, and create corresponding chips for masks. Main packages used were `ee` and `gdal`, with scripts shown in [5_Multi_city_solution _with _GEE](https://github.com/mar-koz22/Park-NET-identifying-Urban-parks-using-multi-source-spatial-data-and-Geo-AI/blob/main/Jiawei's-approach/5_Multi_city_solution_with_GEE.py)

Examples of the whole training process are shown in the folder of [Notebook](https://github.com/mar-koz22/Park-NET-identifying-Urban-parks-using-multi-source-spatial-data-and-Geo-AI/tree/main/Jiawei's-approach/Notebook).

## Main references 
Literatures:

Mapping Urban Green Spaces at the Metropolitan Level Using Very High Resolution Satellite Imagery and Deep Learning Techniques for Semantic Segmentation, https://doi.org/10.3390/rs13112031

An Automatic Extraction Architecture of Urban Green Space Based on DeepLabv3plus Semantic Segmentation Model, https://doi.org/10.1109/ICIVC47709.2019.8981007

Using convolutional networks and satellite imagery to identify patterns in urban environments at a large scale, https://doi.org/10.48550/ARXIV.1704.02965

Codes:

https://segmentation-models.readthedocs.io/en/latest/api.html#unet

https://github.com/bnsreenu/python_for_microscopists

https://geemap.org/notebooks/96_image_chips/
