#!/usr/local/bin/python3
# based on https://fairyonice.github.io/Driver-facial-keypoint-detection-with-data-augmentation.html

import numpy as np
import yaml
from keras.models import Sequential,load_model
from keras.layers import Conv2D, Activation, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, Dense, Flatten
from keras.preprocessing.image import img_to_array, load_img

# Anzahl Durchläufe
epochs = 200

# Trainingsdaten und Bilder aus CSV laden
imgs = []
train_kps = []
with open('data/train/annotations.csv') as f:
    next(f) # erste Zeile überspringen
    for i, line in enumerate(f):
        filename = line.split(';')[0] # erste Spalte (0) = Dateinamen
        data = line.split(';')[1:] # restliche Spalten (1,2,3,..) = Markierungspunkte (key points = kps)
        data = np.asarray(data,dtype=np.float) # array in numpy float array konvertieren

        img = img_to_array(load_img("data/train/"+filename))/255 #Pixel 0-255 -> 0.0-1.0
        imgs.append(img)
        train_kps.append(data)

train_input = np.array(imgs) # numpy array daraus machen...
train_kps = np.array(train_kps)

print (train_input.shape) # Input für das Netz
# (n, 90, 90, 3) => n Bilder mit je 90x90 Pixel x 3 (RGB)

print (train_kps.shape) # Referenzpunkte die gelernt werden sollen
# (n, 10) => Für jedes der n Bilder gibt es 5 Punkte = 10 Werte (2 Augen, 2 Mundwinkel, 1 Nase)


# Normalisieren der Ergenbisswerte
mean_kps = np.mean(train_kps, axis=0) # Mittelwert als neue Null-Linie
std_kps = np.std(train_kps, axis=0) # Standartabweichung als Skalierungsfaktor (-1,+1)

print("Range in original scale: [{:5.3f},{:5.3f})".format(np.min(train_kps), np.max(train_kps)))
train_kps = (train_kps - mean_kps)/std_kps
print("Range in standardized scale: [{:5.3f},{:5.3f})".format(np.min(train_kps), np.max(train_kps)))



### Jetzt das Netz bauen

batch_size = 64

input_shape=train_input[0].shape
result_size=train_kps.shape[1]

model = Sequential()

model.add(Conv2D(16, kernel_size=(5,5),activation='relu',name="CL1",input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

model.add(Conv2D(48, kernel_size=(3, 3),activation='relu',name="CL2"))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',name="CL3"))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

model.add(Conv2D(64,kernel_size=(2, 2),activation='relu',name="CL4"))
model.add(Flatten())

model.add(Dense(100,activation='relu',name="FC5"))
model.add(Dense(10,activation='linear',name="FC6"))

model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])


### training...

hist = model.fit(train_input, train_kps, validation_split=0.2, shuffle=True, epochs=epochs, batch_size=20, verbose=1)



#### Resultat sichern

# speichere trainingsverlauf (loss,val_loss,..) als numpy file
np.save("hist",hist.history)

# speichere modell und berechnete gewichte als HDFS
model.save("model.h5")

# speichere normalisierung daten
np.save('norm',[mean_kps,std_kps])

print("Saved model & training data to disk")
