<h1> Park-NET: identifying Urban parks using multi-source spatial data and Geo-AI </h1>
 
<b>The aim of this project is to analyse to what extent can a reproducible CNN model that predicts urban greenspace based on open source data be created.</b> 
 
The workflow of this project is:
<ul>
<li>data gathering and pre-processing - getting Sentinel satellite images from Earthexplorer, and parks data. Using QGIS to prepare, e.g calculate indices clipping satellite images to city extent, rasterizing park data. </li>
<li>data preparation and augmentation - importing satellite images and corresponding masks (the ground truth for where are parks and where are not), creating chips and balancing data. Balancing was done by deleting chips that have fewer park pixels than the given threshold (e.g. less than 5-20&) Then data augmentation to expand the image dataset. </li>
<li>model training - different models and parameters (e.g. threshold of deleting chips, the stride of the chips, different bands combination) were tested. </li>
<li>evaluation - few cities were left out of the training process to be used as an external validation to access the model accuracy on unseen data.</li>
</ul>
  
I've tried two approaches to solve this:
<ul>
<li> writing U'Net model from scratch, and training it on as many layers as possible. The advantages here are: the model is more understandable and customizable - one can easily add another dropout layer, or change its proportion. Sadly the results are mediocre. </li>
<li> using transfer learning approach, and using pre-trained U-Net with Resnet backbone. This approach is closer to a Black-box approach - one can't edit the model parameters, but it has a backbone that has weights trained on the 2012 ILSVRC ImageNet dataset. The limitation is here that only 3 bands can be used.
</ul>
 
The first approach is in UNet_implemented folder, and the second one is in UNet_with_Resnet_backbone folder.
The second one gave better results, so it will be explained here.


<h2>Transfer learning approach - U-Net with Resnet50 architecture <h2>
 
This approach is in UNet_with_Resnet_backbone folder, and here is explanation of what exactly is done in each file that you can find there.

<h3> 0_create_image_chips_save_numpy_array_github.ipynb </h3> 
This file imports sattelite image, that have 10 bands in total - B, G, R, NIR, SWIR, NDVI, NDBI, NDWI, VARI, land_cover, and park raster. It creates image chips (with patchify library) and saves them as numpy arrays to google drive. It also remove chips with high proportion of backgrounds (over 95-80%) to get balanced training data. 
 
Input - sattelite image and raster with parks:

![image](https://user-images.githubusercontent.com/79871387/168478919-4290f769-7580-440b-be7f-c7b30a6f8901.png)

Output - image patches (saved as numpy arrays):

![image](https://user-images.githubusercontent.com/79871387/168479179-0e84e309-38f9-4c04-b750-185401792654.png)


<h3> 1_UNet_train_model_github.ipynb </h3>
(still in progress)
load numpy arrays with images chips from the google drive and train UNet model on them
<h3> 2_new_city_external_validation_github.ipynb </h3>
(still in progress)
validate model on a new, unseen city
