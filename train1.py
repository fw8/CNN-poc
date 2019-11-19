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


# Trainingsdaten aus CSV laden
train_results = np.loadtxt(open("data/train/annotations.csv", "rb"), delimiter=";", skiprows=1, usecols = (1,2,3,4,5,6,7,8,9,10))

num_lines = train_results.shape[0] # wie viele Zeilen hat das CSV?

train_input = np.array([])
imgs = []
for i in range(num_lines): # jede Zeile ist ein Image -> einlesen
    img = img_to_array(load_img("data/train/image_"+str(i+1)+".jpg"))/255 # pixel 0-255 -> 0.0 - 1.0
    imgs.append(img)

train_input = np.array(imgs) # numpy array daraus machen...

print (train_input.shape) # Input für das Netz
# (n, 90, 90, 3) => n Bilder mit je 90x90 Pixel x 3 (RGB)

print (train_results.shape) # Referenzpunkte die gelernt werden sollen
# (n, 10) => Für jedes der n Bilder gibt es 5 Punkte = 10 Werte (2 Augen, 2 Mundwinkel, 1 Nase)


# Normalisieren der Ergenbisswerte

mY = np.mean(train_results, axis=0) # Mittelwert als neue Null-Linie
sdY = np.std(train_results, axis=0) # Standartabweichung als Skalierungsfaktor (-1,+1)

def standy(x, printTF=False):
    if printTF:
        print("Range in original scale: [{:5.3f},{:5.3f})".format(
            np.min(x), np.max(x)))
    x = (x - mY)/sdY
    if printTF:
        print("Range in standardized scale: [{:5.3f},{:5.3f})".format(
            np.min(x), np.max(x)))
    return(x)

train_results = standy(train_results,True) # normaliseren



### Jetzt das Netz bauen

batch_size = 64
num_channels = 3 # RGB

model = Sequential()

#model.add(BatchNormalization(input_shape=(96, 96, 1)))
model.add(Conv2D(24, 5, 5,
                 border_mode='same',
                 init="he_normal",
                 name="conv2d",
                 input_shape=(90, 90) + (num_channels,)))

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
#model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(500, activation="relu"))
model.add(Dense(90, activation="relu"))
model.add(Dense(10))

model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
model.summary()

### training...

#checkpointer = ModelCheckpoint(filepath="cp_model2.h5", verbose=1, save_best_only=True)

#hist = model.fit(train_input, train_results, validation_split=0.2, shuffle=True, epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)
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
np.save('norm',[mY,sdY])

print("Saved model & training data to disk")
