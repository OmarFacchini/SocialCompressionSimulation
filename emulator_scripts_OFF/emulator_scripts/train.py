import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
#from tensorflow import keras
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import Xception, InceptionV3, DenseNet169
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam




#parser = argparse.ArgumentParser()
#parser.add_argument('--path', default='/media/SSD_new/emu_social', help="path to dataset")
#parser.add_argument('--model', default='/media/SSD_new/emu_social', help="path to dataset")
#parser.add_argument('--out_model', default='/media/SSD_new/emu_social', help="path to dataset")

#args = parser.parse_args()
FLAGS = tf.compat.v1.flags.FLAGS
# dataset
tf.compat.v1.flags.DEFINE_string('path', '/media/SSD_new/emu_social', 'path to dataset')
tf.compat.v1.flags.DEFINE_string('model', '/media/SSD_new/emu_social', 'path to weight')
tf.compat.v1.flags.DEFINE_string('out_model', '/media/SSD_new/emu_social', 'path to output')

#gpus = tf.config.experimental.list_physical_devices('GPU')
#print(gpus)
#tf.config.experimental.set_memory_growth(gpus[0], True)


physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)
with tf.device('/gpu:0'):
    ### DATA GENERATOR
    gen = ImageDataGenerator(
        rescale = 1. / 255.0   # rescaling only
    )

    # path of folders train and test
    train_dir = FLAGS.path + '/train' #'/content/train'
    test_dir = FLAGS.path + '/val'#'/content/test'

    # train generator
    print('Train set generator:')
    train_generator = gen.flow_from_directory(
        train_dir,
        target_size=(224,224),  # (299,299) for Xception and InceptionV3, (224,224) for Dense
        batch_size=32,
        class_mode='categorical',
        classes = {'original':0, 'manipulated':1}   # label 0 to original samples, 1 to manipulated
    )
    print(train_generator)

    # test generator
    print('Test set generator:')
    test_generator = gen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        classes = {'original':0, 'manipulated':1}
    )
    print(test_generator)



    final_model = load_model(FLAGS.model)

    # set trainable layers in final model
    for layer in final_model.layers:
        layer.trainable = True

    # compile
    adam = Adam(learning_rate=0.0002, epsilon=1e-08)
    final_model.compile(loss = 'categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    mcp_save = ModelCheckpoint(FLAGS.out_model, monitor='val_accuracy', save_best_only=True, save_weights_only=False)   # callbacks to save the currently best model
    history = final_model.fit(train_generator, batch_size=32, epochs=15, validation_data=test_generator, verbose=1, callbacks = [mcp_save])



'''
#############################################################
# load base archietecture (InceptionV3)
base_model_In = InceptionV3(weights = 'imagenet', include_top = False)   #load InceptionV3 network pretrained on Imagenet

# riconfigure
x = GlobalAveragePooling2D()(base_model_In.output)
predictions = Dense(2, activation = 'softmax')(x)   # 2 output nodes

# compile
model = Model(base_model_In.input, predictions)

adam = Adam(learning_rate=0.0002, epsilon=1e-08)
model.compile(loss = 'sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

### PRETRAINING
# freeze layers, we train only the last fully connected layer
for layer in model.layers:
    layer.trainable = False


model.fit(train_generator, batch_size=32, epochs = 3, verbose=1, validation_data = test_generator)

# save pretrained model
model.save('weights_inception/pretrained.h5')

### FINAL TRAINING

final_model = load_model('weights_inception/pretrained.h5')

# set trainable layers in final model
for layer in final_model.layers:
    layer.trainable = True

# compile
adam = Adam(learning_rate=0.0002, epsilon=1e-08)
final_model.compile(loss = 'sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

mcp_save = ModelCheckpoint('weights_inception/final_model.h5', monitor='val_accuracy', save_best_only=True, save_weights_only=False)   # callbacks to save the currently best model
history = final_model.fit(train_generator, batch_size=32, epochs=15, validation_data=test_generator, verbose=1, callbacks = [mcp_save])

#############################################################

# load base archietecture (InceptionV3)
base_model_DN = DenseNet169(weights = 'imagenet', include_top = False)   #load Dense network pretrained on Imagenet

# riconfigure
x = GlobalAveragePooling2D()(base_model_DN.output)
predictions = Dense(2, activation = 'softmax')(x)   # 2 output nodes

# compile
model = Model(base_model_DN.input, predictions)

adam = Adam(learning_rate=0.0002, epsilon=1e-08)
model.compile(loss = 'sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

### PRETRAINING
# freeze layers, we train only the last fully connected layer
for layer in model.layers:
    layer.trainable = False


model.fit(train_generator, batch_size=32, epochs = 3, verbose=1, validation_data = test_generator)

# save pretrained model
model.save('weights_densenet/pretrained.h5')

### FINAL TRAINING

final_model = load_model('weights_densenet/pretrained.h5')

# set trainable layers in final model
for layer in final_model.layers:
    layer.trainable = True

# compile
adam = Adam(learning_rate=0.0002, epsilon=1e-08)
final_model.compile(loss = 'sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

mcp_save = ModelCheckpoint('weights_densenet/final_model.h5', monitor='val_accuracy', save_best_only=True, save_weights_only=False)   # callbacks to save the currently best model
history = final_model.fit(train_generator, batch_size=32, epochs=15, validation_data=test_generator, verbose=1, callbacks = [mcp_save])
'''