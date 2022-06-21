# Mapping Public Green Space Using Sentinel-2 Imagery and Convolutional Neural Network at a Global Scale

The main workflow of this project is, 

* First, data preparation, including importing satellite images and corresponding masks (the ground truth for where are parks and where are not), creating chips, balancing data, and data augmentation.

* Second, model building and training. This was done on the 13 chosen cities from different continents: San Francisco, Seattle, Denver, Philadelphia, Greater Manchester, Dublin, Amsterdam, Ghent, Dhaka, Vancouver, Dallas, London, and Buffalo. Both U-Net model from Scratch and with pretrained backbones were tested, with multiple combinations of input bands.

* Third, model validation and prediction on external cities. In this step, models with the best performance would be used to validate on Washington D.C. and Tel Aviv and predict on Kampala.

## Data preparation
Imported satellite images have totally 8 bands and indices, which are R, G, B, NIR, NDVI, NDBI, NDWI, and landcover. Functions defined include `read_file()`, `normalize_by_layer()`, `create_chips()`, `remove_images()`, which are used to import images, normalize each image layer to a range of zero and one, crop image into smaller sizes according to patch size and stride, and remove images with high proportion of backgrounds to balance data, respectively. Python scripts are in [1_Data_preparation](https://github.com/mar-koz22/Park-NET-identifying-Urban-parks-using-multi-source-spatial-data-and-Geo-AI/blob/main/Jiawei's-approach/1_Data_preparation.py). Examples of image chip pairs are: 

<img src="https://user-images.githubusercontent.com/97944674/174822286-3e415b46-2be7-4ee5-8650-494150aaaa2f.png" width="650" height="325">

Techniques chosen for data augmentation are (1) random rotation within an angle of 45 degrees; (2) random width and height shift within a range of 20%; (3) randomly horizontal and vertical flip; (4) randomly zooming in and out within a range of 20%. The python library used was `ImageDataGenerator`. Python scripts are in [2_Data_augmentation](https://github.com/mar-koz22/Park-NET-identifying-Urban-parks-using-multi-source-spatial-data-and-Geo-AI/blob/main/Jiawei's-approach/2_Data_augmentation.py). Example is shown as

<img src="https://user-images.githubusercontent.com/97944674/174823562-00ec884e-40d5-4673-9149-c544d14166fb.png" width="650" height="325">

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

ResNet-50 backbone added much benefit to the model, while VGG-16 made the model perform worse. The converge histories of the three models are plotted below. ResNet-50 helps the training converge faster, however VGG-16 has difficulty to converge within the epochs.

<img src="https://user-images.githubusercontent.com/97944674/174849350-31f35765-8955-4abc-87d4-c12088f73068.png" width="700" height="370">

## Validation & prediction on external cities
U-Net with pretrained ResNet-50 backbone was used to do validation. The average OA, IoU, F-score, and AUC for Washington D.C. are 0.8743, 0.6185, 0.7236, and 	0.7429, for Tel Aviv are 0.8790, 0.4954, 0.5660, and 0.5639, showing a moderate to good generalization capacity of our model.

<img src="https://user-images.githubusercontent.com/97944674/174851740-61a0cffc-40b9-46cb-b9ca-6ec64a8395d5.png" width="700" height="400">

Python scripts to do this are documented in [4_Prediction_and_save_as_tiff](https://github.com/mar-koz22/Park-NET-identifying-Urban-parks-using-multi-source-spatial-data-and-Geo-AI/blob/main/Jiawei's-approach/4_Prediction_and_save_as_tiff.py). Predictions were made for each image chip, and then predicted chips were merged together to a numpy array. This numpy array then was converted into tiff file using given metadata. Main functions used were `unpatchify` in `patchify` library and the defined `array2raster`.

## Multi cities solution
If you are training model on multiple cities, you can use Google Earth Engine to automatically download satellite images, automatically crop satellite images into chips, and create corresponding chips for masks. Main packages used were `ee` and `gdal`, with scripts shown in [5_Multi_city_solution _with _GEE](https://github.com/mar-koz22/Park-NET-identifying-Urban-parks-using-multi-source-spatial-data-and-Geo-AI/blob/main/Jiawei's-approach/5_Multi_city_solution_with_GEE.py)

Reference: 

https://segmentation-models.readthedocs.io/en/latest/api.html#unet

https://github.com/bnsreenu/python_for_microscopists

https://geemap.org/notebooks/96_image_chips/
