# Park-NET: identifying Urban parks using multi source spatial data and Geo-AI
Different models and parameters (e.g., image size, combinations of bands) were tested.
## 1_U-net_ResNet_backbone
A ResNet34 backbone is used as the encoder part of U-Net model. As the backbone only accepts three-channel images (otherwise you cannot make use of the pretrained weights or you need to reshape your images into 3-channel), three bands were chosen among all the 8 bands. Different combinations were tested.
### Land surface temperature (LST) + NDVI + land cover using 256\*256 image chips
The model was training on San Francisco, Seattle, and Denver, training history is plotted below:

<img src="https://user-images.githubusercontent.com/97944674/168816303-b124de7f-252a-4758-93f3-7c99a4d7937b.png" width="300" height="200"><img src="https://user-images.githubusercontent.com/97944674/168817375-9b5a8ce3-6b56-4033-8c4d-85d685ba472b.png" width="300" height="200"><img src="https://user-images.githubusercontent.com/97944674/168817858-ff956bc0-b040-430d-8430-5db0e2b23bd8.png" width="300" height="200">

The overall accuracy of this model is 0.965, with IoU 0.912, F1 score 0.953, AUC 0.950. When testing on a new city, Philadelphia, the overall accuracy of this model is 0.829, with IoU 0.619, F1 score 0.745, AUC 0.746. Some random images were tested:

<img src="https://user-images.githubusercontent.com/97944674/168819399-d7701385-ebda-47de-baf5-bac039d13462.png" width="800" height="250">

