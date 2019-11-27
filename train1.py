#!/usr/local/bin/python3

# Model idea from:
# https://towardsdatascience.com/detecting-facial-features-using-deep-learning-2e23c8660a7a

import numpy as np
import yaml
from keras.models import Sequential,load_model
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten
from keras.preprocessing.image import img_to_array, load_img

# Anzahl Durchläufe
epochs = 300

# Trainingsdaten und Bilder aus CSV laden
imgs = []
train_results = []
with open('data/train/annotations.csv') as f:
    next(f)
    for i, line in enumerate(f):
        filename = line.split(';')[0] # erste Spalte (0) = Dateinamen
        data = line.split(';')[1:] # restliche Spalten (1,2,3,..) = Markierungspunkte
        data = np.asarray(data,dtype=np.float) # array in numpy float array konvertieren

        img = img_to_array(load_img("data/train/"+filename))/255 #Pixel 0-255 -> 0.0-1.0
        imgs.append(img)
        train_results.append(data)

train_input = np.array(imgs) # numpy array daraus machen...
train_results = np.array(train_results)

print (train_input.shape) # Input für das Netz
# (n, 90, 90, 3) => n Bilder mit je 90x90 Pixel x 3 (RGB)

print (train_results.shape) # Referenzpunkte die gelernt werden sollen
# (n, 10) => Für jedes der n Bilder gibt es 5 Punkte = 10 Werte (2 Augen, 2 Mundwinkel, 1 Nase)


# Normalisieren der Ergenbisswerte
mean_results = np.mean(train_results, axis=0) # Mittelwert als neue Null-Linie
std_results = np.std(train_results, axis=0) # Standartabweichung als Skalierungsfaktor (-1,+1)


def normalize(x, show=False):
    if show:
        print("Range in original scale: [{:5.3f},{:5.3f})".format(np.min(x), np.max(x)))
    x = (x - mean_results)/std_results
    if show:
        print("Range in standardized scale: [{:5.3f},{:5.3f})".format(np.min(x), np.max(x)))
    return(x)

train_results = normalize(train_results,True) # normaliseren



### Jetzt das Netz bauen

batch_size = 64

input_shape=train_input[0].shape
result_size=train_results.shape[1]

model = Sequential()

model.add(Conv2D(24, 5, 5,
                 border_mode='same',
                 init="he_normal",
                 name="conv2d",
                 input_shape=input_shape))

model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
model.add(Conv2D(36, (5, 5)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
model.add(Conv2D(48, (5, 5)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dense(500, activation="relu"))
model.add(Dense(90, activation="relu"))
model.add(Dense(result_size))

model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
model.summary()


### training...

hist = model.fit(train_input, train_results, validation_split=0.2, shuffle=True, epochs=epochs, batch_size=20, verbose=1)



#### Resultat sichern

# serialize model to YAML
model_yaml = model.to_yaml()
with open("model1.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)

# speichere trainingsverlauf (loss,val_loss,..) als yaml file
hist_yaml = yaml.dump(hist.history)
with open("hist1.yaml", "w") as yaml_file:
    yaml_file.write(hist_yaml)

# speichere modell und berechnete gewichte als HDFS
model.save("model1.h5")

# speichere normalisierung daten
np.save('norm',[mean_results,std_results])

print("Saved model & training data to disk")
