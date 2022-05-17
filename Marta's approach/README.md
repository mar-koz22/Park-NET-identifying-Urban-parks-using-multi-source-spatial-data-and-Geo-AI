<h1> Park-NET: identifying Urban parks using multi source spatial data and Geo-AI </h1>

<h3> 0_create_image_chips_save_numpy_array_github.ipynb </h3> 
This file takes sattelite image and park raster and creates chips (with patchify library) and saves them as numpy arrays to google drive.
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