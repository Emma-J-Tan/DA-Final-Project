# -*- coding: utf-8 -*-

## 1.0 Intro

- **Problem**: training a deep CNN from scratch to distinguish HLD.


<span style="color:red">**If you are running this on Colab, make sure you enable GPU!!!**</span>

## 2.0 Environment

### 2.1 Unzip data

- Please first download the data from the github link above as its directory structure has been set up already.
- If you are running this notebook on Colab, you need to upload the zip file to Colab environment then proceed to the next step.
- If you are running this notebook on your local server, simply put the zip file in the same directory as your notebook.
"""

import zipfile

unzip_target_dir = '.'
with zipfile.ZipFile('./HLDimages.zip', 'r') as zip_ref:
  zip_ref.extractall(unzip_target_dir)

# Comment out this entire block if you are running locally.
from google.colab import files

"""### 2.2 Define constants for training, validation, and test data

"""

import os

base_dir = './HLD_NonHLD';

train_dir = os.path.join(base_dir, 'train')
train_hld_dir = os.path.join(train_dir, 'hld')
train_nonhld_dir = os.path.join(train_dir, 'nonhld')

validation_dir = os.path.join(base_dir, 'validation')
validation_hld_dir = os.path.join(validation_dir, 'hld')
validation_nonhld_dir = os.path.join(validation_dir, 'nonhld')

test_dir = os.path.join(base_dir, 'test')
test_hld_dir = os.path.join(test_dir, 'hld')
test_nonhld_dir = os.path.join(test_dir, 'nonhld')

print('Folder structure:', os.listdir('./hld_and_nonhld_small'))
print('Total traing hld images:', len(os.listdir(train_hld_dir)))
print('Total traing nonhld images:', len(os.listdir(train_nonhld_dir)))
print('Total validation hld images:', len(os.listdir(validation_hld_dir)))
print('Total validation nonhld images:', len(os.listdir(validation_nonhld_dir)))
print('Total test hld images:', len(os.listdir(test_hld_dir)))
print('Total test nonhld images:', len(os.listdir(test_nonhld_dir)))

"""## 3.0 Naive CNN

### 3.1 Model
"""

from keras import layers, models, optimizers

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc']
)

model.summary()

"""### 3.2 Data Preprocessing


1. Read the source files (images).
2. Decode the JPEG content to RGB grids of pixels.
3. Convert these into floating-point tensors.
4. Rescale the pixel values from a value in [0, 255] to a value in [0, 1].

Documentation on ImageDataGenerator: https://keras.io/preprocessing/image/
"""

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

for data_batch, labels_batch in train_generator:
  print('data batch shape:', data_batch.shape)
  print('labels batch shape:', labels_batch.shape)
  break

"""### 3.3 Training

Again, if you are on Colab, make sure you have enabled GPU -- it's a matter of minutes vs. hours.
"""

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100, # batch_size = 20 => 100 batch = 2000 samples
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

model.save('hld_and_nonhld_small_naive_cnn.h5')

# If you are on Colab and want to download the file, uncomment this
# files.download('hld_and_nonhld_small_naive_cnn.h5')

"""### 3.4 Plotting"""

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'm', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'm', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

"""## 4.0 Improvement

### 4.1 Data Augmentation

We augment the samples via a number of random transformation that yield believable-looking images. This helps expose the model to more aspects of the data and in turn makes it generalize better.
"""

train_datagen = ImageDataGenerator(
    rescale=1./255,         # - rescale
    rotation_range=40,      # - degrees
    width_shift_range=0.2,  # - ranges as a fraction of total width or height
    height_shift_range=0.2, #   to randomly translate
    shear_range=0.2,        # - shearing
    zoom_range=0.2,         # - zooming
    horizontal_flip=True,   # - horizontal flip
    fill_mode='nearest'     # - strategy to fill the newly created pixels
)

# We are not augmenting test data
test_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

datagen = ImageDataGenerator(
    rotation_range=40,      # - degrees
    width_shift_range=0.2,  # - ranges as a fraction of total width or height
    height_shift_range=0.2, #   to randomly translate
    shear_range=0.2,        # - shearing
    zoom_range=0.2,         # - zooming
    horizontal_flip=True,   # - horizontal flip
    fill_mode='nearest'     # - strategy to fill the newly created pixels
)

"""### 4.2 Adding Dropouts

To further fight overfitting, we add a Dropout layer right before the densely connected classifier.
"""

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc']
)

model.summary()

"""### 4.3 Train"""

from keras.callbacks import ModelCheckpoint
import time

# Use checkpoints here as it takes a bit longer
filepath="./weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath,
    monitor='val_acc',
    verbose=1,
    save_best_only=True,
    mode='max',
    period=5 # Checkpoints after each 5 iterations
)

callbacks_list = [checkpoint]

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50,
    callbacks=callbacks_list
)

model.save('hld_and_nonhld_small_improved.h5')

# Uncomment if you want to download the model
# files.download('hld_and_nonhld_small_improved.h5')

"""### 4.4 Plotting"""

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'm', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'm', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

"""## 5.0 Pretrained Model

### 5.1 VGG16

The VGG16 architecture is a simple and widely used convnet architecture for ImageNet. We will use its convolutional base and train our own dense classifier on top of it.
"""

from keras.applihldions import VGG16

conv_base = VGG16(
    weights='imagenet',  # weight checkpoint from which we initialize the model
    include_top=False,    # discard the classifier, only conv base is needed
    input_shape=(150, 150, 3)
)

# Build the new model by adding a dense layer on top of conv_base
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

"""### 5.2 Freeze Base and Train Classifier

We freeze the conv base so that back-prop won't destroy its already-learned weight.
"""

conv_base.trainable = False

"""### 5.3 Train the model

Note that the training data has already been augmented from above.
"""

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=2e-5),
    metrics=['acc']
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

model.save('hld_and_nonhld_small_pretrained.h5')

"""### 5.4 Plotting"""

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'm', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'm', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

"""### 5.5 Fine-tuning

Now we unfreeze the last three layers of conv base and train them along with the classifier.
"""

conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
  if layer.name == 'block5_conv1':
    set_trainable = True
  if set_trainable:
    layer.trainable = True
  else:
    layer.trainable=False

model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-5),
    metrics=['acc']
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50
)

model.save('hld_and_nonhld_full_pretrained_final.h5')

"""### 5.6 Plotting"""

files.download('hld_and_nonhld_full_pretrained_final.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'm', label='Training Accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'm', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

"""### 5.7 Make Predictions

Finally, we use the test set to measure the accuracy of our final model -- around 97%.
"""

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print("test acc:", test_acc)
