<h1> PPark-NET: Identifying Public Urban Green Spaces Using Multi-Source Spatial Data and Convolutional Networks </h1>
 
<b>The aim of this project is to analyse to what extent can a reproducible CNN model that predicts public urban green spaces (PUGSs) based on open source data be created.</b> 
 
The workflow of this project is:
<ul>
<li>data gathering and pre-processing - getting Sentinel satellite images from Earthexplorer, and parks data. Using QGIS to prepare, e.g calculate indices clipping satellite images to city extent, rasterizing park data. </li>
<li>data preparation and augmentation - importing satellite images and corresponding masks (the ground truth for where are parks and where are not), creating chips and balancing data. Balancing was done by deleting chips that have fewer park pixels than the given threshold (less than 10%) Then data augmentation to expand the image dataset. </li>
<li>model training - different models and parameters (e.g. threshold of deleting chips, the stride of the chips, different bands combination) were tested. </li>
<li>evaluation - few cities were left out of the training process to be used as an external validation to access the model accuracy on unseen data.</li>
</ul>
  
I've tried two approaches to solve this:
<ul>
<li> writing U-Net model from scratch. The advantages here are: the model is more understandable and customizable - one can easily add another dropout layer, or change its proportion. Sadly the results are mediocre. </li>
<li> using transfer learning approach, and using pre-trained U-Net with ResNet34 backbone. This approach is closer to a Black-box approach - one can't edit the model parameters, but it has a backbone that has weights trained on the 2012 ILSVRC ImageNet dataset. The limitation is here that only 3 bands can be used.
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

The full methodological workflow:
<img src="https://user-images.githubusercontent.com/79871387/175558186-2383e3d5-1c83-4bb3-ace9-f06686488143.jpg">

This process was mainly done in Google Clab Pro, and scripts are descibed here:

<h3> 0_create_image_chips_save_numpy_array_github.ipynb </h3> 

[0_create_image_chips_save_numpy_array_github.ipynb](https://github.com/mar-koz22/Park-NET-identifying-Urban-parks-using-multi-source-spatial-data-and-Geo-AI/blob/main/Marta's-approach/0_create_image_chips.ipynb) file imports sattelite image, that have 8 bands in total - B, G, R, NIR, NDVI, NDBI, NDWI, land_cover, and park raster. It creates image chips with [Patchify library](https://pypi.org/project/patchify/) and saves them as numpy arrays to google drive. It also remove chips with high proportion of backgrounds (over 90%) to get balanced training data. 
 
Input - sattelite image and raster with parks:

<img src="https://user-images.githubusercontent.com/79871387/168478919-4290f769-7580-440b-be7f-c7b30a6f8901.png" width="650">

Output - image patches (saved as numpy arrays):

<img src="https://user-images.githubusercontent.com/79871387/168479179-0e84e309-38f9-4c04-b750-185401792654.png" width="700">

<h2>Transfer learning approach - U-Net with ResNet34 architecture </h2>

<h3> 1b_UNet_train_model_github.ipynb </h3>

[1b_UNet_train_model_github.ipynb](https://github.com/mar-koz22/Park-NET-identifying-Urban-parks-using-multi-source-spatial-data-and-Geo-AI/blob/main/Marta's-approach/UNet_with_Resnet_backbone/1b_UNet_parks_with_a_backbone_github.ipynb) file is training the prediction model. It starts with reading image and mask chips from 10 different cities from one of the 9 three-band compositions. Cities that were included in the training process are Amsterdam, Buffalo, Dhaka, Dublin, Ghent, London, Manchester, Philadelphia, Seattle, Vancouver. After dividing into train and test, and preprocessing data to fit the backbone architecture, data augumentation using ImageDataGenerator library is done to make the dataset more diverse.

Data augumentation examples:

<img src="https://user-images.githubusercontent.com/79871387/175539657-ae314c94-7006-4458-bf3a-6f9d1b088832.png" width="700">

Next step is training the model. Pre-trained model was implemented from [Segmentations Models](https://github.com/qubvel/segmentation_models) library.


<h3> 2b_new_city_external_validation_github.ipynb </h3>

[2b_test_on_new_city_backbone_github](https://github.com/mar-koz22/Park-NET-identifying-Urban-parks-using-multi-source-spatial-data-and-Geo-AI/blob/main/Marta's-approach/UNet_with_Resnet_backbone/2b_test_on_new_city_backbone_github.ipynb) is checking the accuracy on external data. Two cities were used for the external validation - Washington and Tel Aviv. Sattelite image and mask are loaded, and chips are created. Public urban green spaces are predicted.

<h3> 3_produce_image_output.ipynb </h3>

[3_produce_image_output.ipynb](https://github.com/mar-koz22/Park-NET-identifying-Urban-parks-using-multi-source-spatial-data-and-Geo-AI/blob/main/Marta's-approach/3_produce_image_output.ipynb) produces tiff image with public urban green spaces predictions based on the chosen model.

<h2>Results</h2>

<h3> Training process </h3>
Transfer learning is helping with making the training process converge faster. For U-Net with ResNet34 encoder the learning process is also less bumpy then for model from scratch, the loss and IoU are changing more smoothly..

<img src="https://user-images.githubusercontent.com/79871387/175571701-584615aa-e241-4149-b7fe-a4c74cd9bfbf.png" width="650">

<h3> External validation accuracy </h3>
All the presented results are based on two external cities – Washington and Tel Aviv. Both of those cities weren’t used for training and testing of any of the models. The IoU, and F1 of the semantic segmentation done by the U-Net with ResNet34 backbone for each of the 9 three-band compositions calculated based on 2 external validation cities are presented in table.

![image](https://user-images.githubusercontent.com/79871387/175573898-f4693e3c-a88d-411d-b84a-d26bca3ea114.png)

Comparison of PUGSs prediction in Washington. Left is ground truth, middle the best U-Net with ResNet34 encoder based on Red-NIR-NDVI, right best model from scratch, so Red-NDWI-Landcover. PUGS are white, and background is black

![image](https://user-images.githubusercontent.com/79871387/175574332-c603340f-5322-4ad8-98c6-b7249a7d53f5.png)

<h3> Prediction for Washington </h3>
After evaluating all models, the best model was chosen - Red-NIR-NDVI, U-Net with a ResNet34 encoder. This model was used to create new PUGSs datasets for 3 external cities. Here is presented Washington. Right - ground truth PUGS data for Washington, left predicted PUGS data for Washington. PUGS are green, background is white:

![image](https://user-images.githubusercontent.com/79871387/175574651-ff5b2dd6-6721-4688-a1d6-8d9fcfef7120.png)

When looking up close at the prediction there are a few groups of misclassifications when comparing prediction with ground truth data for Washington:
1. Small PUGSs in high density neighbourhoods. Figure shows comparison of true colour Washington image (left) and PUGSs prediction on top of ground truth data (right). These examples show that when there are small PUGSs in a dense neighbourhood model sometimes fails or predicts just parts of the PUGS. Left true colour Washington image Right satellite image, on top of that PUGSs predictions as green,
and ground truth PUGSs symbolised with cross filling.

![image](https://user-images.githubusercontent.com/79871387/175575272-77a344bf-fe69-47b5-a760-56a4c23e5fc2.png)

2. Some parts of PUGSs that are build-ups not green space
Figure shows example when part of a PUGS is some infrastructure, building - build-up area, not green. Then created model will likely predict that those parts are background not PUGS. Left true colour Washington image Right satellite image, on top of that PUGSs predictions as green,
and ground truth PUGSs symbolised with cross filling.

![image](https://user-images.githubusercontent.com/79871387/175575378-1759d24b-3f46-44bd-adb2-5fbaefa4782d.png)

3. Green spaces that were not in the ground truth data, that are probably not public. Figure shows example of predicted area that is probably a golf club but was predicted as a PUGS. Left true colour Washington image Right satellite image, on top of that PUGSs predictions as green,
and ground truth PUGSs symbolised with cross filling.

![image](https://user-images.githubusercontent.com/79871387/175575478-917e500f-03df-4b2a-9898-aa54e54f15b9.png)

<h3> Prediction for Tel Aviv </h3>

![image](https://user-images.githubusercontent.com/79871387/175576652-849a43f9-9583-4071-aaf1-62319871e695.png)


<h3> Prediction for Kampala </h3>

![image](https://user-images.githubusercontent.com/79871387/175576505-af08d5ec-45de-4742-a062-ef4c057880df.png)
