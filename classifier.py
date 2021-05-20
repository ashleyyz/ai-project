import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

# Download and explore dataset
import pathlib
dataset_url = "https://github.com/ashleyyz/ai-project/raw/master/images.tar"
data_dir = tf.keras.utils.get_file('images', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# Create a dataset
batch_size = 100
img_height = 300
img_width = 300

# 80/20 validation split
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Print class names
class_names = train_ds.class_names
print(class_names)

    
# Training batches
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Normalize data
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# Create Model
num_classes = 3

data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# Compile model
model.compile(optimizer= Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train Model
epochs = 75
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Visualize training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Save Model
model.save("saved_model")
del model

model = load_model('saved_model')

tennis_url = "https://github.com/ashleyyz/ai-project/raw/master/tennis-racket.jpg"
tennis_path = tf.keras.utils.get_file('tennis-racket', origin=tennis_url)
baddy_url = "https://github.com/ashleyyz/ai-project/raw/master/badminton-racket.jpg"
baddy_path = tf.keras.utils.get_file('badminton-racket', origin=baddy_url)
squash_url = "https://github.com/ashleyyz/ai-project/raw/master/squash-racket.jpg"
squash_path = tf.keras.utils.get_file('squash-racket', origin=squash_url)

tennis_img = keras.preprocessing.image.load_img(
    tennis_path, target_size=(img_height, img_width)
)
tennis_img_array = keras.preprocessing.image.img_to_array(tennis_img)
tennis_img_array = tf.expand_dims(tennis_img_array, 0) # Create a batch

baddy_img = keras.preprocessing.image.load_img(
    baddy_path, target_size=(img_height, img_width)
)
baddy_img_array = keras.preprocessing.image.img_to_array(baddy_img)
baddy_img_array = tf.expand_dims(baddy_img_array, 0) # Create a batch

squash_img = keras.preprocessing.image.load_img(
    squash_path, target_size=(img_height, img_width)
)
squash_img_array = keras.preprocessing.image.img_to_array(squash_img)
squash_img_array = tf.expand_dims(squash_img_array, 0) # Create a batch

tennis_predictions = model.predict(tennis_img_array)
tennis_score = tf.nn.softmax(tennis_predictions[0])
print(
    "This is a {} racket with a {:.2f} percent confidence."
    .format(class_names[np.argmax(tennis_score)], 100 * np.max(tennis_score))
)

baddy_predictions = model.predict(baddy_img_array)
baddy_score = tf.nn.softmax(baddy_predictions[0])
print(
    "This is a {} racket with a {:.2f} percent confidence."
    .format(class_names[np.argmax(baddy_score)], 100 * np.max(baddy_score))
)

squash_predictions = model.predict(squash_img_array)
squash_score = tf.nn.softmax(squash_predictions[0])
print(
    "This is a {} racket with a {:.2f} percent confidence."
    .format(class_names[np.argmax(squash_score)], 100 * np.max(squash_score))
) 