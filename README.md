# DeepLearningStudy

## [View in Colaboratory](https://colab.research.google.com/github/vinimyls/DeepLearningStudy/blob/master/DeepLearning.ipynb)

## Imports


```python
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
```

### Loading dataset


```python
dataset = keras.datasets.fashion_mnist
((traine_picture, traine_id), (test_picture, test_id)) = dataset.load_data()

```

### Data exploration


```python
len(traine_picture)
traine_picture.shape
test_picture.shape
len(test_id)
traine_id.min()
traine_id.max()
```

### Data exhibition


```python
total_of_kinds = 10
name_of_kinds = ['T-shirt', 'Pants', 'Pullover',
                            'Dress', 'Coat', 'Sandal', 'Shirt',
                            'Tennis', 'Bag', 'Boot']
plt.imshow(traine_picture[0])
plt.colorbar()
```

### Normalizing the pictures


```python
traine_picture = traine_picture/float(255)
```

### Creat, compiling, training and normalizing the model


```python
model = keras.Sequential([ 
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(256, activation=tensorflow.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tensorflow.nn.softmax)
])

model.compile(optimizer='adam', 
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])

historic = model.fit(traine_picture, traine_id, epochs=5, validation_split=0.2)
```

### Saving and loading the model


```python
model.save('model.h5')
model_save = load_model('model.h5')
```

### Viewing training and validation accuracy by season


```python
plt.plot(historic.history['accuracy'])
plt.plot(historic.history['val_accuracy'])
plt.title('Accuracy per epoch')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['traine', 'validation'])

```

### Viewing training losses and validation by season


```python
plt.plot(historic.history['loss'])
plt.plot(historic.history['val_loss'])
plt.title('Loss per epoch')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['traine', 'validation'])
```

### Testing the model and the saved model


```python
tests = model.predict(test_picture)
print('test result:', np.argmax(tests[1]))
print('número da imagem de teste:', test_id[1])

test_model_save = model_save.predict(test_picture)
print('resultado teste modelo salvo:', np.argmax(test_model_save[1]))
print('número da imagem de teste:', test_id[1])
```

### Evaluating the model


```python
test_loss, test_accuracy = model.evaluate(test_picture, test_id)
print('Perda do teste:', test_loss)
print('Acurácia do teste:', test_accuracy)
```
