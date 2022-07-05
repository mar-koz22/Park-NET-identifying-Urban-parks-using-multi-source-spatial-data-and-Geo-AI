<h1> Park-NET: Identifying Public Urban Green Spaces Using Multi-Source Spatial Data and Convolutional Networks </h1>
 
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
![project workflow (7)](https://user-images.githubusercontent.com/79871387/177307484-dc29e254-4f4d-4323-acb1-6bb34bd5fe00.jpg)


This process was mainly done in Google Clab Pro, and scripts are descibed here:

<h3> 0_create_image_chips_save_numpy_array_github.ipynb </h3> 

[0_create_image_chips_save_numpy_array_github.ipynb](https://github.com/mar-koz22/Park-NET-identifying-Urban-parks-using-multi-source-spatial-data-and-Geo-AI/blob/main/Marta's-approach/0_create_image_chips.ipynb) file imports sattelite image, that have 8 bands in total - B, G, R, NIR, NDVI, NDBI, NDWI, land_cover, and park raster. It creates image chips with [Patchify library](https://pypi.org/project/patchify/) and saves them as numpy arrays to google drive. It also remove chips with high proportion of backgrounds (over 90%) to get balanced training data. 
 
Input - sattelite image and raster with parks:

<img src="https://user-images.githubusercontent.com/79871387/168478919-4290f769-7580-440b-be7f-c7b30a6f8901.png" width="650">

Output - image patches (saved as numpy arrays):

<img src="https://user-images.githubusercontent.com/79871387/168479179-0e84e309-38f9-4c04-b750-185401792654.png" width="700">

<h2>Transfer learning approach - U-Net with ResNet34 architecture </h2>

<h3> 1a_UNet_resnet34_encoder.ipynb </h3>

[1a_UNet_resnet34_encoder.ipynb](https://github.com/mar-koz22/Park-NET-identifying-Urban-parks-using-multi-source-spatial-data-and-Geo-AI/blob/main/Marta's-approach/1a_UNet_resnet34_encoder.ipynb) file is training the prediction model. It starts with reading image and mask chips from 10 different cities from one of the 9 three-band compositions. Cities that were included in the training process are Amsterdam, Buffalo, Dhaka, Dublin, Ghent, London, Manchester, Philadelphia, Seattle, Vancouver. After dividing into train and test, and preprocessing data to fit the backbone architecture, data augumentation using ImageDataGenerator library is done to make the dataset more diverse.

Data augumentation examples:

<img src="https://user-images.githubusercontent.com/79871387/175539657-ae314c94-7006-4458-bf3a-6f9d1b088832.png" width="700">

Next step is training the model. Pre-trained model was implemented from [Segmentations Models](https://github.com/qubvel/segmentation_models) library.


<h3> 2a_evaluation_UNet_resnet34_encoder.ipynb </h3>

[2a_evaluation_UNet_resnet34_encoder.ipynb](https://github.com/mar-koz22/Park-NET-identifying-Urban-parks-using-multi-source-spatial-data-and-Geo-AI/blob/main/Marta's-approach/2a_evaluation_UNet_resnet34_encoder.ipynb) is checking the accuracy on external data. Two cities were used for the external validation - Washington and Tel Aviv. Sattelite image and mask are loaded, and chips are created. Public urban green spaces are predicted.

<h3> 3_produce_image_output.ipynb </h3>

[3_produce_image_output.ipynb](https://github.com/mar-koz22/Park-NET-identifying-Urban-parks-using-multi-source-spatial-data-and-Geo-AI/blob/main/Marta's-approach/3_produce_image_output.ipynb) produces tiff image with public urban green spaces predictions based on the chosen model.

<h2>Results</h2>

<h3>Training, validation results and choosing the best model</h3>

Training and validation results are presented for all the created models. Training set was 80% of the input data, and validation was 20%. There were 18 models created in total, because there were 9 different three-band compositions, and two different model architectures – baseline U-Net model from scratch and U-Net model with a pre-trained ResNet34 encoder. All the models were trained on 10 cities mentioned in Table 2. The aim was to find the best combination of model architecture and band composition based on validation metrics.

First for baseline model:

![image](https://user-images.githubusercontent.com/79871387/177301627-ede99336-ae79-4356-99cf-6e570c45f1ea.png)

Then for model with ResNet34 backbone:

![image](https://user-images.githubusercontent.com/79871387/177301942-08e43fae-b013-452f-a542-2fb7415c6798.png)

For baseline U-Net from scratch average validation IoU was 0,7259, average validation F1 score was 0,8403. For U-Net with ResNet34 encoder it was 0,8681 and 0,9274, which is a 12% rise for validation IoU and 8% for validation F1 score. This visible rise shows that U-Net with a ResNet34 backbone achieved better performance than the baseline model, and that transfer learning is a very good approach. The best model should be chosen from the models with a pre-trained encoder.


NIR-NDWI-NDBI composition with U-Net with ResNet34 encoder architecture achieved the best average validation score among all the models and was chosen as the best model to be used later for evaluation and predictions. It had validation IoU of 0,8770, and validation F1 score of 0,9326.

<h3> Training process </h3>
Transfer learning is helping with making the training process converge faster. For U-Net with ResNet34 encoder the learning process is also less bumpy then for model from scratch, the loss and IoU are changing more smoothly.

<img src="https://user-images.githubusercontent.com/79871387/175571701-584615aa-e241-4149-b7fe-a4c74cd9bfbf.png" width="650">

<h2> External validation accuracy </h2>
External evaluation was based on test set that consisted of two external cities – Washington and Tel Aviv. Both of those cities were not a part of testing or validation process. Test performance of the chosen model, so the NIR-NDWI-NDBI U-Net with a ResNet34 encoder is:

![image](https://user-images.githubusercontent.com/79871387/177306582-c35ceba2-fe11-48d0-9e96-ce3485fad205.png)

The best model achieved an average test IoU of 0,5610, and average test F1 score of 0,64515 across two external cities. Washington had an average of 0,6622 test metrics, and Tel Aviv had 0,544, so the model performed visibly better on Washington then on Tel Aviv.

<h3> Prediction for Washington using NIR-NDWI-NDBI U-Net with a ResNet34 encoder model</h3>

Differences between validation metrics of baseline models from scratch and models with a backbone were visible, but when we compare their prediction on external dataset this difference in performance is even more apparent. Figure belowe illustrates differences in prediction of Washington PUGSs between chosen NIR-NDWI-NDBI U-Net with a Resnet34 encoder, and model from scratch that achieved the best validation metrics, so NDVI-NDWI-Landcover baseline model. It is evident that the transfer learning model is performing a lot better on an external test set.

![image](https://user-images.githubusercontent.com/79871387/177306916-9503f2bf-8d95-471e-9bf6-c416d1b36565.png)


The best, chosen model, so NIR-NDWI-NDBI U-Net with a Resnet34 encoder was used to create new PUGSs datasets for 3 external cities – Washington, Tel Aviv, and Kampala. 

Here is presented Washington. Right - ground truth PUGS data for Washington, left predicted PUGS data for Washington. PUGS are green, background is white:

![image](https://user-images.githubusercontent.com/79871387/175574651-ff5b2dd6-6721-4688-a1d6-8d9fcfef7120.png)

Looking at the predictions for Washington there are some parts, like the big green space in the north, and longitudinal green space in the middle, that were predicted good, but there are also some misclassifications that should be analysed up close. 


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

<h3> Prediction for Tel Aviv using NIR-NDWI-NDBI U-Net with a Resnet34 encoder model</h3>

![image](https://user-images.githubusercontent.com/79871387/175576652-849a43f9-9583-4071-aaf1-62319871e695.png)


<h3> Prediction for Kampala using NIR-NDWI-NDBI U-Net with a Resnet34 encoder model</h3>

![image](https://user-images.githubusercontent.com/79871387/175576505-af08d5ec-45de-4742-a062-ef4c057880df.png)

<h4> Reference </h4>
Coding:
<ul>
<li>Segmentation Models library - https://segmentation-models.readthedocs.io/en/latest/index.html & https://github.com/qubvel/segmentation_models </li>
<li>Patchify library - https://pypi.org/project/patchify/ & https://github.com/dovahcrow/patchify.py </li>
<li>https://github.com/bnsreenu/python_for_microscopists </li>
<li>https://github.com/jordancaraballo/nga-deep-learning </li>
</ul>
 Main literature:
<ul>
<li>Using Convolutional Networks and Satellite Imagery to Identify Patterns in Urban Environments at a Large Scale by Albert, A., Kaur, J., & Gonzalez, M. (2017) - https://arxiv.org/abs/1704.02965 </li>
<li>Mapping Urban Green Spaces at the Metropolitan Level Using Very High Resolution Satellite Imagery and Deep Learning Techniques for Semantic Segmentation by Huerta, R. E., Yépez, F. D., Lozano-García, D. F., Guerra Cobián, V. H., Ferriño Fierro, A. L., de León Gómez, H., Cavazos González, R. A., & Vargas-Martínez, A. (2021) - https://www.mdpi.com/2072-4292/13/11/2031 </li>
 <li>U-Net: Convolutional Networks for Biomedical Image Segmentation by Ronneberger, O., Fischer, P., & Brox, T. (2015) - https://arxiv.org/abs/1505.04597

