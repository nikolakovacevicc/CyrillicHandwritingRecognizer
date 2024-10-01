
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import random

import glob
import os

import tensorflow as tf
import matplotlib.pyplot as plt
import imageio
import cv2
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import plot_model
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, LeakyReLU,GlobalMaxPooling2D
from tensorflow.keras.metrics import top_k_categorical_accuracy,categorical_accuracy
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping



# putanja do slika
PATH_TO_DATA = './dataset/'
data_info =  ['letters.csv', 'letters2.csv', 'letters3.csv']

#Ucitavanje csv fajlova
def import_data(CSV_FILE):
    data = pd.read_csv(PATH_TO_DATA + CSV_FILE)
    data['source'] = CSV_FILE[:-4]+'/'
    return data


data1 = import_data(data_info[0])
data2 = import_data(data_info[1])
data3 = import_data(data_info[2])
data = pd.concat([data1, data2, data3], ignore_index=True)

del(data1, data2, data3)

data = shuffle(data, random_state = 42)

print(data.head())#prikaz prvih par redova (provere radi)

#Razlicita slova
letters = ''
for letter in data.letter.unique():
    letters += letter
labele = data.label




#OH labels
def ohe_letter(label):
    result = np.zeros(len(letters))
    index = letters.index(label)
    result[index] = 1
    return result

#OH backgrounds
def ohe_background(label):
    result = np.zeros(len(data.background.unique()))
    result[label] = 1
    return result

#novi data
data['enc_letter'] = data['letter'].apply(ohe_letter)
data['enc_background'] = data['background'].apply(ohe_background)
print(data.head())

#dimenzije slika
img_width, img_height = 32, 32
input_shape = (img_width, img_height, 3)

images = [] #skup gde cemo staviti slike
encoded_labels = []#ciljani izlazi slika


backgrounds = []
encoded_backgrounds = []
niz={}

#ucitavnje slika i pozadina i njihovih izlaz
for i, row in data.iterrows():
    img_name = row['file']
    numpy_image = cv2.imread(os.path.join(PATH_TO_DATA +row['source'], img_name))
    if numpy_image.shape == (32, 32, 3):
        images.append(numpy_image)
        encoded_labels.append(row['enc_letter'])
        backgrounds.append(row['background'])
        encoded_backgrounds.append(row['enc_background'])
        if row['label'] not in niz:
            niz[row['label']] = 0
        niz[row['label']] += 1


#normalizacija
images = np.array(images)/255

print('Broj ucitanih slika: '+str(len(images)))

plt.figure()
plt.hist(data['label'], bins=len(letters), color='red', edgecolor='black', alpha=0.5,rwidth=0.5)
plt.show()
print(niz)

#stampanje primera svake klase
def display_images_and_labels(images, labels):
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    labels = labels.tolist()
    for label in unique_labels:
        # uzimamo prvu sliku za svaku od labela
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    plt.show()


display_images_and_labels(images, np.argmax(encoded_labels, axis=1))


ulaz = np.array(images.copy())
izlazOH = np.array(encoded_labels.copy())

ulazTrening, ulazTest, izlazTreningOH, izlazTestOH = train_test_split(ulaz, izlazOH,
                                                                      stratify = izlazOH,
                                                                      test_size=0.2,
                                                                      random_state=42)

ulazTrening, ulazVal, izlazTreningOH, izlazValOH = train_test_split(ulazTrening,izlazTreningOH,
                                                                    stratify = izlazTreningOH,
                                                                    train_size=0.8,
                                                                    random_state=42)


print(ulazTrening.shape)
print(ulazVal.shape)
print(ulazTest.shape)

print(izlazTreningOH.shape)
print(izlazValOH.shape)
print(izlazTestOH.shape)


from keras import layers
'''
data_augmentation = Sequential(
 [
 layers.RandomFlip("horizontal", input_shape=input_shape),
 layers.RandomRotation(0.25),
 layers.RandomZoom(0.1),
 ]
)

model = Sequential([
 data_augmentation,
 layers.Conv2D(32, 3, padding='same', activation='relu'),
 #layers.MaxPooling2D(pool_size = (2, 2)),
 layers.Conv2D(64, 3, padding='same', activation='relu'),
 layers.MaxPooling2D(pool_size = (2, 2)),
 layers.Conv2D(128, 3, padding='same', activation='relu'),
 layers.MaxPooling2D(pool_size = (2, 2)),
 layers.Dropout(0.25),
 layers.Flatten(),
 layers.Dense(128, activation='relu'),
 layers.Dropout(0.25),
 layers.Dense(len(letters), activation='softmax')
])'''



# dimenzije ulaza u mrezu
input_shape = (img_width, img_height, 3)

# broj klasa
num_classes = len(letters)

batch_size = 64

# broj epoha
epochs = 150


# Zbog velike kolicine klasa lakse dolazi do greske pa cemo pratiti i top_3 metriku
def top_3_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


data_augmentation = Sequential(
 [
 layers.RandomFlip("horizontal", input_shape=input_shape),
 layers.RandomRotation(0.25),
 layers.RandomZoom(0.1),
 ]
)


from keras.regularizers import l2

model = Sequential([
 data_augmentation,
 layers.Conv2D(32, kernel_size = (3, 3), padding='same', activation='relu'),
 #layers.MaxPooling2D(pool_size = (2, 2)),
 layers.Conv2D(64, (3,3), padding='same', activation='relu'),
 layers.MaxPooling2D(pool_size = (2, 2)),
 layers.Conv2D(128, (4,4), padding='same', activation='relu'),
 layers.MaxPooling2D(pool_size = (2, 2)),
 layers.Dropout(0.25),
 layers.Flatten(),
 layers.Dense(128, activation='relu'),
 layers.Dropout(0.25),
 layers.Dense(len(letters), activation='softmax')
])


early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy', top_3_categorical_accuracy])
model.summary()


lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                                 patience=10,
                                 verbose=2,
                                 factor=.75)




model_checkpoint= ModelCheckpoint("./best_result_checkpoint", monitor='val_loss', save_best_only=True, verbose=0)




history = model.fit(ulazTrening,izlazTreningOH,
                    batch_size = batch_size,
                    epochs = epochs,
                    verbose = 1,
                    validation_data = (ulazVal,izlazValOH),
                    callbacks = [model_checkpoint, lr_reduction,early_stopping])

'''loss, acc,top3_cat_acc= cnn_model.evaluate(ulazTest, izlazTestOH)
print("loss", loss)
print("acc", acc)
print("top 3 category acc", top3_cat_acc)'''

plt.plot(history.history['accuracy'],'--', label='accuracy on training set')
plt.plot(history.history['val_accuracy'], label='accuracy on validation set')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

cnn_model=models.load_model("./best_result_checkpoint",
                            custom_objects={'top_3_categorical_accuracy':top_3_categorical_accuracy})
cnn_model.summary()



dobri=np.array([])
losi=np.array([])

dobri_pred=np.array([])
losi_pred=np.array([])

dobri_target=np.array([])
losi_target=np.array([])

labels = np.array([])
pred = np.array([])
for i in range(ulazTest.shape[0]):
    img=ulazTest[i]
    img=np.expand_dims(img, axis=0)
    lab=izlazTestOH[i]
    l=np.argmax(lab)
    labels = np.append(labels, l)
    p=np.argmax(cnn_model.predict(img, verbose=0), axis=1)
    pred = np.append(pred, p)
    if( l == p ):
        dobri = np.append(dobri, img)
        dobri_pred=np.append(dobri_pred,p)
        dobri_target = np.append(dobri_target, l)
    else:
        losi = np.append(losi, img)
        losi_pred = np.append(losi_pred, p)
        losi_target = np.append(losi_target, l)

from sklearn.metrics import accuracy_score

print('Tačnost modela je: ' + str(100 * accuracy_score(labels, pred)) + '%')

print(dobri.shape)
print(losi.shape)

dobri=dobri.reshape((-1, 32, 32, 3))
losi=losi.reshape((-1, 32, 32, 3))

print(dobri.shape)
print(losi.shape)
# Kreiramo subplot
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 5))

# Prolazimo kroz elemente prvog niza i prikazujemo ih na prvom redu subplota
for i in range(5):
    axs[0, i].imshow(dobri[i])
    axs[0,i].set_title('dobra procena '+letters[int(dobri_target[i])]+'('+letters[int(dobri_pred[i])]+')')
    axs[0, i].axis('off')

# Prolazimo kroz elemente drugog niza i prikazujemo ih na drugom redu subplota
for i in range(5):
    axs[1, i].imshow(losi[i])
    axs[1,i].set_title('losa procena '+letters[int(losi_target[i])]+'('+letters[int(losi_pred[i])]+')')
    axs[1, i].axis('off')

# Prikazujemo plot
plt.show()






class_labels = [str(i) for i in range(num_classes)]
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(labels, pred, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels )
cmDisplay.plot()
plt.show()


labels1 = np.array([])
pred1 = np.array([])
for i in range(ulazTrening.shape[0]):
    img=ulazTrening[i]
    img=np.expand_dims(img, axis=0)
    lab=izlazTreningOH[i]
    labels1 = np.append(labels, np.argmax(lab))
    pred1 = np.append(pred, np.argmax(cnn_model.predict(img, verbose=0), axis=1))




from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm1 = confusion_matrix(labels1, pred1, normalize='true')
cmDisplay1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=class_labels )
cmDisplay1.plot()
plt.show()


from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.losses import BinaryCrossentropy
from keras.callbacks import EarlyStopping
'''
model.summary()
model.compile(Adam(learning_rate=0.001),
 loss='categorical_crossentropy',
 metrics='accuracy')

es = EarlyStopping(monitor='val_loss', mode='min', patience=20, verbose=1)

lr_reduction = ReduceLROnPlateau(monitor='val_loss',
                                 patience=10,
                                 verbose=2,
                                 factor=.75)

model_checkpoint= ModelCheckpoint("./best_result_checkpoint", monitor='val_loss', save_best_only=True, verbose=0)

history = model.fit(ulazTrening,izlazTreningOH,
                    batch_size=64,
                    epochs=100,
                    validation_data=(ulazVal,izlazValOH),
                    verbose=0,
                    #callbacks = [model_checkpoint, lr_reduction]
                    callbacks=[es]
                    )


plt.plot(history.history['accuracy'],'--', label='accuracy on training set')
plt.plot(history.history['val_accuracy'], label='accuracy on validation set')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#cnn_model=models.load_model("./best_result_checkpoint")
#cnn_model.summary()


labels = np.array([])
pred = np.array([])
for i in range(ulazTest.shape[0]):
    img=ulazTest[i]
    img=np.expand_dims(img, axis=0)
    lab=izlazTestOH[i]
    labels = np.append(labels, np.argmax(lab))
    pred = np.append(pred, np.argmax(model.predict(img, verbose=0), axis=1))

from sklearn.metrics import accuracy_score

print('Tačnost modela je: ' + str(100 * accuracy_score(labels, pred)) + '%')
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(labels, pred, normalize='true')
cmDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=len(letters))
cmDisplay.plot()
plt.show()
'''

