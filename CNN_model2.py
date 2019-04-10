from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, BatchNormalization
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.callbacks import History
from keras import regularizers
import os, sys
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

def normalize(X_all):
    # Feature normalization with train and test X
    X_train_test = X_all
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    return X_train_test

#f = open(sys.argv[1],'r')
f = open('./datas/train.csv', 'r')
datas = f.readlines()[1:]
Y_train = [] 
X_train = []
n = 0
for data in datas:
    label, pixels = data.split(',')
    pixels = list(map(float,pixels.split(' ')))
    if np.mean(pixels) <= 40 or np.mean(pixels) >= 215:
        continue
    X_train.append([])
    X_train[n] = pixels
    Y_train.append(float(label))
    n += 1
del datas


#### end 
X_train = np.array(X_train) / 255.

X_train = X_train.reshape(len(X_train), 48, 48, 1)
Y_train = np.array(Y_train)

Y_train = np.eye(7)[list(map(int,Y_train))]

### validation set
from sklearn.model_selection import train_test_split
X_train, x_val, Y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2)

# Add Gaussian noise 3000 pics
px = np.random.randint(0, len(X_train), size=5000)
temp = X_train[px] + np.random.normal(0.0, 0.2, size=(len(px), 48, 48, 1))
X_train = np.append(X_train, temp, axis=0)
Y_train = np.append(Y_train, Y_train[px], axis=0)


## Normalize
X_train = X_train.reshape(len(X_train), 2304)
X_train = normalize(X_train)
X_train = X_train.reshape(len(X_train), 48, 48, 1)

x_val = x_val.reshape(len(x_val), 2304)
x_val = normalize(x_val)
x_val = x_val.reshape(len(x_val), 48, 48, 1)

## Data generator
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(X_train)

## Initialize
model =  Sequential()
## add CNN
model.add(Conv2D(64, (3,3), input_shape=(48,48,1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
#model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
#model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(256, (1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
#model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Conv2D(512, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
#model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Flatten())

## add DNN
model.add(Dense(300, use_bias=False, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(128, use_bias=False, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(7, use_bias=True, activation = 'softmax'))

## compile loss function
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()

### checkpoint function
checkpoint = ModelCheckpoint("output/best.h5",monitor='val_acc',verbose=0, save_best_only=True,mode='max')
### call back function
call = EarlyStopping(monitor='val_acc', min_delta=0, patience=50)
call_back_list = [checkpoint, call]

### fit
#history = model.fit(X_train, Y_train, validation_data=(x_val,y_val), batch_size=128, epochs=350, callbacks=call_back_list, shuffle=True)
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=128),
        steps_per_epoch=int(len(X_train)/128),
        validation_data=(x_val, y_val), validation_steps=1,
        epochs=100,
        callbacks=call_back_list)

### plot_model_acc
import csv
acc = history.history['acc']
val_acc = history.history['val_acc']
text = open('output/CNN_acc.csv','w')
s = csv.writer(text,delimiter=',',lineterminator='\n')
for i in range(len(acc)):
    s.writerow([acc[i],val_acc[i]]) 
text.close()
###

f2 = open("./datas/test.csv", 'r')
datas = f2.readlines()[1:]
X_test = []
n = 0
for data in datas:
    X_test.append([])
    id, pixels = data.split(',')
    pixels = np.array(list(map(float, pixels.split(' '))))
    #X_test[n] = pixels.reshape((48,48,1))
    X_test[n] = pixels
    n += 1

X_test = np.array(X_test) / 255.
X_test = normalize(X_test)
X_test = X_test.reshape(7178,48,48,1)

## Predict
results = model.predict(X_test)
print(results.shape)
ans =[]
ans.append(['id','label'])
for i in range(results.shape[0]):
  ans.append([i,np.argmax(results[i,:])])

with open('output/result.csv','w+') as f:
  s = csv.writer(f,delimiter=',',lineterminator='\n')
  for i in range(len(ans)):
   s.writerow(ans[i])

from keras.models import load_model
model.save('output/my_model.h5')

