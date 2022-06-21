# build model (U-Net with resnet50 backbone, pretrained on imagenet)
import segmentation_models as sm
import keras
import tensorflow
from keras.layers import Input, Conv2D
from keras.models import Model

n_classes = 2
activation = 'sigmoid'  # the activation function used for the output layer

# set loss function and metrics
LR = 0.0001
optim = tensorflow.keras.optimizers.Adam(LR)
loss = sm.losses.binary_focal_dice_loss
metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# set BACKBONE
BACKBONE = 'resnet50'  # other backbones can be used here, e.g., vgg16

# preprocess X_train and y_train to fit the model architecture
preprocess_input = sm.get_preprocessing(BACKBONE) # get the preprocessing function
X_train = preprocess_input(X_train)  # load X_train
X_test = preprocess_input(X_test)   # load X_test

# define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=n_classes, activation=activation)
# you can also use other 3 model architectures, Linknet, FPN, and PSPNet

# compile keras model with defined optimizer, loss and metrics
model.compile(optim, loss = loss, metrics=metrics)
model.summary()

# set hyperparameters
batch_size = 16
steps_per_epoch = len(X_train)//batch_size  # for generator, you need to specify the steps if using generator
validation_steps = len(X_test)//batch_size  # for generator
print(steps_per_epoch, validation_steps)

# model training (with data augmentation)
history = model.fit(train_generator, batch_size=16, epochs=50,  steps_per_epoch = steps_per_epoch,
                    validation_steps = validation_steps, verbose=1, validation_data = validation_generator)
