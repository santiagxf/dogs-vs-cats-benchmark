import os, cv2, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
from random import shuffle 
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D

if ('__file__' in globals()):
    working_dir = os.path.dirname(os.path.abspath(__file__))
else:
    working_dir = r'C:\Users\facun\Projects\Kaggle\dogs-vs-cats-benchmark'


IMG_SIZE = 224
NUM_CLASSES = 2
SAMPLE_SIZE = 20000
NO_EPOCHS = 10
BATCH_SIZE = 64
RANDOM_STATE = 2018
TEST_SIZE = 0.25
TRAIN_FOLDER = os.path.join(working_dir, 'data\\train')
TEST_FOLDER =  os.path.join(working_dir, 'data\\test')


def labelPetImageOneHotEncoder(img):
    pet = img.split('.')[-3]
    if pet == 'cat': return [1,0]
    elif pet == 'dog': return [0,1]

def processData(data_image_list, DATA_FOLDER, isTrain=True):
    data_df = []
    for img in tqdm(data_image_list):
        path = os.path.join(DATA_FOLDER,img)
        if(isTrain):
            label = labelPetImageOneHotEncoder(img)
        else:
            label = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        data_df.append([np.array(img),np.array(label)])
    shuffle(data_df)
    return data_df

def createModel():
    base_model = ResNet50(include_top=False, pooling='max', weights='imagenet')

    model = Sequential()
    model.add(base_model)
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def createModelV2():
    base_model = ResNet50(include_top=False, pooling='max', weights='imagenet')

    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(200, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


train_image_list = os.listdir(TRAIN_FOLDER)
test_image_list = os.listdir(TEST_FOLDER)
train = processData(train_image_list, TRAIN_FOLDER)
test = processData(test_image_list, TEST_FOLDER, False)

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
y = np.array([i[1] for i in train])

model = createModel()
model.summary()

plot_model(model, to_file='model.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

train_model = model.fit(X_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=NO_EPOCHS,
                  verbose=1,
                  validation_data=(X_val, y_val))

score = model.evaluate(X_val, y_val, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])


#get the predictions for the test data
predicted_classes = model.predict_classes(X_val)
#get the indices to be plotted
y_true = np.argmax(y_val,axis=1)

correct = np.nonzero(predicted_classes==y_true)[0]
incorrect = np.nonzero(predicted_classes!=y_true)[0]

target_names = ["Class {}:".format(i) for i in range(NUM_CLASSES)]
print(classification_report(y_true, predicted_classes, target_names=target_names))

