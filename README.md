# Importing the libraries


```python
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
```

# Part 1 - Data Preprocessing

## Preprocessing the Training set


```python
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
```

    Found 1234 images belonging to 2 classes.
    

## Preprocessing the Test set


```python
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set', target_size = (64, 64), batch_size = 32, class_mode = 'binary')
```

    Found 218 images belonging to 2 classes.
    

# Part 2 - Building the CNN Model

## Initializing the CNN Model


```python
model = tf.keras.models.Sequential()
```

## Step 1 - Convolution


```python
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
```

## Step 2 - Pooling


```python
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
```

## Adding a second convolutional layer


```python
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
```

## Step 3 - Flattening


```python
model.add(tf.keras.layers.Flatten())
```

## Step 4 - Full Connection


```python
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
```

## Step 5 - Output Layer


```python
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
```

# Part 3 - Training the CNN Model

## Compiling the CNN Model


```python
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
```

## Training the CNN Model on the Training set and evaluating it on the Test set


```python
model.fit(x = training_set, validation_data = test_set, epochs = 25)
```

    Epoch 1/25
    39/39 [==============================] - 8s 210ms/step - loss: 0.4420 - accuracy: 0.7925 - val_loss: 0.1767 - val_accuracy: 0.9450
    Epoch 2/25
    39/39 [==============================] - 9s 219ms/step - loss: 0.1450 - accuracy: 0.9449 - val_loss: 0.1240 - val_accuracy: 0.9633
    Epoch 3/25
    39/39 [==============================] - 8s 213ms/step - loss: 0.1070 - accuracy: 0.9579 - val_loss: 0.1160 - val_accuracy: 0.9679
    Epoch 4/25
    39/39 [==============================] - 8s 212ms/step - loss: 0.0850 - accuracy: 0.9635 - val_loss: 0.1617 - val_accuracy: 0.9587
    Epoch 5/25
    39/39 [==============================] - 8s 205ms/step - loss: 0.0879 - accuracy: 0.9611 - val_loss: 0.1050 - val_accuracy: 0.9679
    Epoch 6/25
    39/39 [==============================] - 8s 205ms/step - loss: 0.0571 - accuracy: 0.9797 - val_loss: 0.1022 - val_accuracy: 0.9725
    Epoch 7/25
    39/39 [==============================] - 8s 205ms/step - loss: 0.0594 - accuracy: 0.9797 - val_loss: 0.0964 - val_accuracy: 0.9541
    Epoch 8/25
    39/39 [==============================] - 8s 206ms/step - loss: 0.0688 - accuracy: 0.9749 - val_loss: 0.1332 - val_accuracy: 0.9679
    Epoch 9/25
    39/39 [==============================] - 8s 206ms/step - loss: 0.0672 - accuracy: 0.9724 - val_loss: 0.1422 - val_accuracy: 0.9633
    Epoch 10/25
    39/39 [==============================] - 8s 205ms/step - loss: 0.0831 - accuracy: 0.9684 - val_loss: 0.1136 - val_accuracy: 0.9679
    Epoch 11/25
    39/39 [==============================] - 8s 206ms/step - loss: 0.0472 - accuracy: 0.9846 - val_loss: 0.1015 - val_accuracy: 0.9633
    Epoch 12/25
    39/39 [==============================] - 8s 204ms/step - loss: 0.0500 - accuracy: 0.9838 - val_loss: 0.0982 - val_accuracy: 0.9725
    Epoch 13/25
    39/39 [==============================] - 8s 206ms/step - loss: 0.0465 - accuracy: 0.9830 - val_loss: 0.1211 - val_accuracy: 0.9725
    Epoch 14/25
    39/39 [==============================] - 8s 206ms/step - loss: 0.0398 - accuracy: 0.9862 - val_loss: 0.0861 - val_accuracy: 0.9679
    Epoch 15/25
    39/39 [==============================] - 8s 206ms/step - loss: 0.0402 - accuracy: 0.9830 - val_loss: 0.0964 - val_accuracy: 0.9679
    Epoch 16/25
    39/39 [==============================] - 8s 218ms/step - loss: 0.0167 - accuracy: 0.9959 - val_loss: 0.0929 - val_accuracy: 0.9679
    Epoch 17/25
    39/39 [==============================] - 8s 204ms/step - loss: 0.0443 - accuracy: 0.9854 - val_loss: 0.1038 - val_accuracy: 0.9679
    Epoch 18/25
    39/39 [==============================] - 8s 203ms/step - loss: 0.0336 - accuracy: 0.9887 - val_loss: 0.0797 - val_accuracy: 0.9725
    Epoch 19/25
    39/39 [==============================] - 8s 200ms/step - loss: 0.0376 - accuracy: 0.9878 - val_loss: 0.1229 - val_accuracy: 0.9679
    Epoch 20/25
    39/39 [==============================] - 8s 206ms/step - loss: 0.0437 - accuracy: 0.9854 - val_loss: 0.1220 - val_accuracy: 0.9725
    Epoch 21/25
    39/39 [==============================] - 8s 204ms/step - loss: 0.0323 - accuracy: 0.9862 - val_loss: 0.1144 - val_accuracy: 0.9725
    Epoch 22/25
    39/39 [==============================] - 8s 206ms/step - loss: 0.0373 - accuracy: 0.9854 - val_loss: 0.1075 - val_accuracy: 0.9633
    Epoch 23/25
    39/39 [==============================] - 8s 206ms/step - loss: 0.0391 - accuracy: 0.9854 - val_loss: 0.0891 - val_accuracy: 0.9725
    Epoch 24/25
    39/39 [==============================] - 8s 205ms/step - loss: 0.0431 - accuracy: 0.9846 - val_loss: 0.1008 - val_accuracy: 0.9679
    Epoch 25/25
    39/39 [==============================] - 8s 205ms/step - loss: 0.0444 - accuracy: 0.9862 - val_loss: 0.1023 - val_accuracy: 0.9633
    




    <tensorflow.python.keras.callbacks.History at 0x255af740940>



# Part 4 - Making a single prediction


```python
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/closed_or_open_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'open'
else:
    prediction = 'closed'
print(prediction)
```

    closed
    


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 62, 62, 32)        896       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 31, 31, 32)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 29, 29, 32)        9248      
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 14, 14, 32)        0         
    _________________________________________________________________
    flatten (Flatten)            (None, 6272)              0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               802944    
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 129       
    =================================================================
    Total params: 813,217
    Trainable params: 813,217
    Non-trainable params: 0
    _________________________________________________________________
    
