# import libraries
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, Dropout
from keras.layers import Activation, MaxPool2D, Concatenate

def conv_block(input, num_filters):
  '''
  Each conV block consists of two convolutional layer.
  Each convolutional layer consists of one convolutional operation (with size 3 * 3),
  one normalization operation, one dropout layer, and one activation operation ("ReLU").
  '''
  x = Conv2D(num_filters, 3, padding="same")(input)
  x = BatchNormalization()(x)
  x = Dropout(0.2)(x)
  x = Activation("relu")(x)

  x = Conv2D(num_filters, 3, padding="same")(x)
  x = BatchNormalization()(x)
  x = Dropout(0.2)(x)
  x = Activation("relu")(x)

  return x

# Encoder block
def encoder_block(input, num_filters):
  '''
  Encoder block consists of a conv_block and one maxpooling
  x: output of conv_block
  p: output after maxplooling
  '''
  x = conv_block(input, num_filters)
  p = MaxPool2D((2, 2))(x)
  return x, p

# Decoder block
def decoder_block(input, skip_features, num_filters):
  '''
  Decoder block consists of an upsampling operation, a concatenate operation, and one convolutional block
  Inputs: the output of previous layer, skip feature, and number of filters
  Skip features are the output from encoder for concatenation
  Skip feature will be concatenated with output of unsampling operation
  '''
  x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
  x = Concatenate()([x, skip_features])
  x = conv_block(x, num_filters)
  return x

# Build Unet using the blocks
def park_unet(input_shape):
  inputs = Input(input_shape)

  s1, p1 = encoder_block(inputs, 64)  # number of filters can be customized
  s2, p2 = encoder_block(p1, 128)
  s3, p3 = encoder_block(p2, 256)
  s4, p4 = encoder_block(p3, 512)

  b1 = conv_block(p4, 1024) #Bridge

  d1 = decoder_block(b1, s4, 512)
  d2 = decoder_block(d1, s3, 256)
  d3 = decoder_block(d2, s2, 128)
  d4 = decoder_block(d3, s1, 64)

  outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)  # the output will be a one-channel image since it's a binary segmentation
  model = Model(inputs, outputs, name="U-Net")
  return model
