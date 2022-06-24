<h1> PPark-NET: Identifying Public Urban Green Spaces Using Multi-Source Spatial Data and Convolutional Networks </h1>
 
<b>The aim of this project is to analyse to what extent can a reproducible CNN model that predicts urban greenspace based on open source data be created.</b> 
 
The workflow of this project is:
<ul>
<li>data gathering and pre-processing - getting Sentinel satellite images from Earthexplorer, and parks data. Using QGIS to prepare, e.g calculate indices clipping satellite images to city extent, rasterizing park data. </li>
<li>data preparation and augmentation - importing satellite images and corresponding masks (the ground truth for where are parks and where are not), creating chips and balancing data. Balancing was done by deleting chips that have fewer park pixels than the given threshold (e.g. less than 5-20%) Then data augmentation to expand the image dataset. </li>
<li>model training - different models and parameters (e.g. threshold of deleting chips, the stride of the chips, different bands combination) were tested. </li>
<li>evaluation - few cities were left out of the training process to be used as an external validation to access the model accuracy on unseen data.</li>
</ul>
  
I've tried two approaches to solve this:
<ul>
<li> writing U-Net model from scratch. The advantages here are: the model is more understandable and customizable - one can easily add another dropout layer, or change its proportion. Sadly the results are mediocre. </li>
<li> using transfer learning approach, and using pre-trained U-Net with Resnet backbone. This approach is closer to a Black-box approach - one can't edit the model parameters, but it has a backbone that has weights trained on the 2012 ILSVRC ImageNet dataset. The limitation is here that only 3 bands can be used. Pre-trained model was implemented from [Segmentations Models](https://github.com/qubvel/segmentation_models) library.
</ul>

So those were the the model archcitectures that this study evaluated.

Because litearature suggest this approach, and the pre-trained can take 3 bands 9 three-band compositions were choosen:
<ul>
<li>Blue, Green, Red </li>
<li>Green, Red, NIR</li>
<li>Red, NIR, NDVI</li>
<li>NDVI, NDBI, Landcover</li>
<li>Red, NDWI, Landcover</li>
<li>NDVI, NDWI Landcover</li>
<li>NIR, NDWI, NDBI</li>
<li>NDBI, NDWI, Landcover</li>
<li>NIR, NDWI, Landcover</li>
</ul>

So this study created 18 models - 2 models architectures and 9 band compositions.
 
The first approach is in [UNet_implemented](https://github.com/mar-koz22/Park-NET-identifying-Urban-parks-using-multi-source-spatial-data-and-Geo-AI/tree/main/Marta's-approach/UNet_implemented) folder, and the second one is in [UNet_with_Resnet_backbone](https://github.com/mar-koz22/Park-NET-identifying-Urban-parks-using-multi-source-spatial-data-and-Geo-AI/tree/main/Marta's-approach/UNet_with_Resnet_backbone) folder.
The second one gave better results, so it will be explained here.


<h2>Transfer learning approach - U-Net with Resnet50 architecture </h2>

<h3> 0b_create_image_chips_save_numpy_array_github.ipynb </h3> 

[0b_create_image_chips_save_numpy_array_github.ipynb](https://github.com/mar-koz22/Park-NET-identifying-Urban-parks-using-multi-source-spatial-data-and-Geo-AI/blob/main/Marta's-approach/UNet_with_Resnet_backbone/0b_create_image_chips_save_numpy_array_github.ipynb) file imports sattelite image, that have 8 bands in total - B, G, R, NIR, NDVI, NDBI, NDWI, land_cover, and park raster. It creates image chips with [Patchify library](https://pypi.org/project/patchify/) and saves them as numpy arrays to google drive. It also remove chips with high proportion of backgrounds (over 90%) to get balanced training data. 
 
Input - sattelite image and raster with parks:

![image](https://user-images.githubusercontent.com/79871387/168478919-4290f769-7580-440b-be7f-c7b30a6f8901.png)

Output - image patches (saved as numpy arrays):

![image](https://user-images.githubusercontent.com/79871387/168479179-0e84e309-38f9-4c04-b750-185401792654.png)


<h3> 1b_UNet_train_model_github.ipynb </h3>
(still in progress)
This file is training the prediction model. It starts with reading image and mask chips from different cities - currently Amsterdam, Dublin, Ghent, Manchester, Seattle and San Francisco with image chips that have no less then 10% of parks. This image chips have 3 bands - Ndvi, Landcover, Ndwi. After dividing into train and test, and preprocessing data to fit the backbone architecture, data augumentation using ImageDataGenerator library is done to make the dataset more diverse.

Next step is training the model. For now the best accuracy was achieved using ResNet50 backbone, 50 epochs, and including chips that have no less then 10% of park. 

![image](https://user-images.githubusercontent.com/79871387/172175542-9455bb59-2ecb-4bc7-8b08-5e6e97f375b9.png)

When training on Amsterdam, Dublin, Ghent, Manchester, Seattle and San Francisco, the model has a overall accuracy of 0.96, IoU of 0.87.

![image](https://user-images.githubusercontent.com/79871387/172175945-aa16cbf6-1e67-44cd-a4ba-3dc10e95ff1d.png)


<h3> 2b_new_city_external_validation_github.ipynb </h3>
(still in progress)

This file is checking the accuracy on external data. Here city of Philadephia was used, because it wasn't part of the trining process. Sattelite image and mask is loaded,and chips are created. Parks are predicted, and the accuracy of the prediction in 0.87, and IOU is 0.59.

![image](https://user-images.githubusercontent.com/79871387/172181598-df2aec14-3c43-4b63-b3ec-a7ed9a0ddcf1.png)

