import os

import cv2
import keras
import numpy as np
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from keras.layers import SeparableConv2D
from keras.models import Model
from keras.optimizers import Adam
from utils import rescale_box

Beta=[0.011676873, 0.869083715, 0.119239412]
Beta= np.array(Beta)

def convert_to_logits(y_pred):
    # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    return tf.math.log(y_pred / (1 - y_pred))

def blance_loss(y_true, y_pred):
    y_pred = convert_to_logits(y_pred)
    pos_weight =Beta
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)

    # or reduce_sum and/or axis=-1
    return tf.reduce_mean(loss *(Beta))


#Model
inputs = Input((256,256,3))
conv1 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
conv1 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = SeparableConv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
conv4 = SeparableConv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
conv5 = SeparableConv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5 = SeparableConv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
drop5 = Dropout(0.5)(conv5)
up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
merge6 = concatenate([drop4,up6], axis = 3)
conv6 = SeparableConv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6 = SeparableConv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
merge7 = concatenate([conv3,up7], axis = 3)
conv7 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7 = SeparableConv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
merge8 = concatenate([conv2,up8], axis = 3)
conv8 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8 = SeparableConv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
merge9 = concatenate([conv1,up9], axis = 3)
conv9 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9 = SeparableConv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv9 = SeparableConv2D(3, 3, activation = 'softmax', padding = 'same', kernel_initializer = 'he_normal')(conv9)

model = Model(inputs,conv9)
model.compile(optimizer = Adam(lr = 1e-4), loss =blance_loss, metrics = ['accuracy'])


model.load_weights(os.path.join('SG_model', 'model.h5'))


def predict_(input_img):
    original = np.array(cv2.imread(str(input_img)))
    resized_image = cv2.resize(original, (256, 256), interpolation=cv2.INTER_NEAREST)
    input_ = resized_image.reshape(1, 256, 256, 3)
    input_ = input_/255
    pre = model.predict(input_)
    softmax_output = pre.reshape(256, 256, 3)
    pre = np.argmax(softmax_output, axis=-1)
    label = keras.utils.to_categorical(pre, 3)
    label[:, :, 0] = label[:, :, 1]
    label[:, :, 2] = label[:, :, 1]
    label = label*255
    label = label.astype('float32')
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(label, 127, 255, 0)
    thresh = thresh.astype(np.uint8)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes_original = list()
    bounding_boxes_rescaled = list()
    trash_masks = list()
    for contour in contours:
        mask = np.zeros_like(resized_image)
        cv2.fillPoly(mask, pts=[contour], color=(1, 1, 1))
        trash_masks.append(mask)
        # Mutiply mask with softmax output and sum over the output to add all the probabilities and then divide by
        # total number of pixels in one channel (divide by 3 since channels are 3) to get average probability
        trash_prob = np.sum((mask * softmax_output)[:, :, 1]) / (np.count_nonzero(mask) / 3)
        (x, y, w, h) = cv2.boundingRect(contour)
        xx = x + w
        yy = y + w
        bounding_boxes_original.append([trash_prob, x, y, xx, yy])
        box_coordinates = [x, y, xx, yy]
        x, y, xx, yy = rescale_box(original, resized_image, box_coordinates)
        bounding_boxes_rescaled.append([trash_prob, x, y, xx, yy])

    return trash_masks, bounding_boxes_original, bounding_boxes_rescaled


if __name__ == '__main__':
    predict_('test.jpg')