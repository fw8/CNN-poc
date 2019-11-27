#!/usr/local/bin/python3
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

# lade fertiges modell und gewichte
model = load_model('model1.h5')
# summarize model.
model.summary()


# lade test daten
# Trainingsdaten und Bilder aus CSV laden
imgs = []
test_results = []
with open('data/test/annotations.csv') as f:
    next(f)
    for i, line in enumerate(f):
        filename = line.split(';')[0] # erste Spalte (0) = Dateinamen
        data = line.split(';')[1:] # restliche Spalten (1,2,3,..) = Markierungspunkte
        data = np.asarray(data,dtype=np.float) # array in numpy float array konvertieren

        img = img_to_array(load_img("data/test/"+filename))/255 #Pixel 0-255 -> 0.0-1.0
        imgs.append(img)
        test_results.append(data)

test_input = np.array(imgs) # numpy array daraus machen...
test_results = np.array(test_results)


print (test_input.shape) # Input für das Netz
# (n, 90, 90, 3) => n Bilder mit je 90x90 x3(RGB)

print (test_results.shape) # Referenz Werte gegen die das Netz antreten muss
# (n, 10) => Für jedes der n Bilder gibt es 5 Punkte = 10 Werte (2 Augen, 2 Mundwinkel, 1 Nase)


# evaluate the model (noch nicht geguckt was hier genau gemacht wird...)
score = model.evaluate(test_input, test_results, verbose=1)
print (score)


# und jetzt werden die testdaten ins netz gefüttert...
pred_results = model.predict(test_input)


# entnormalisieren
#
# laden der normalisierungswerte (mittelwert u. standartabweichung)
# der trainingsdaten
norm = np.load('norm.npy',allow_pickle=True)
mean_results = norm[0]
std_results = norm[1]

# normalisierung rückgängig machen...
pred_results = pred_results * std_results + mean_results



# show i. image
i = 10

img = test_input[i]
#cv2.imshow("image", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

plt.imshow(img)

# facial key points
p1 = test_results[i]

plt.scatter(p1[0],p1[1], c="red") # Rechtes Auge
plt.scatter(p1[2],p1[3], c="green") # Linkes Auge
plt.scatter(p1[4],p1[5], c="blue") # Nase
plt.scatter(p1[6],p1[7], c="yellow") # Rechter Mundwinckel
plt.scatter(p1[8],p1[9], c="orange") # Linker Mundwinkel

p1 = pred_results[i]

plt.scatter(p1[0],p1[1], c="red", marker='X') # Rechtes Auge
plt.scatter(p1[2],p1[3], c="green", marker='X') # Linkes Auge
plt.scatter(p1[4],p1[5], c="blue", marker='X') # Nase
plt.scatter(p1[6],p1[7], c="yellow", marker='X') # Rechter Mundwinckel
plt.scatter(p1[8],p1[9], c="orange", marker='X') # Linker Mundwinkel

plt.show()
