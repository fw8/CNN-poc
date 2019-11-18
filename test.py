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
test_results = np.loadtxt(open("data/test/annotations.csv", "rb"), delimiter=";", skiprows=1, usecols = (1,2,3,4,5,6,7,8,9,10))

num_lines = test_results.shape[0] # wie viele Zeilen hat das CSV?

test_input = np.array([])
imgs = []
for i in range(num_lines): # jede Zeile ist ein Image -> einlesen
    img = img_to_array(load_img("data/test/image_"+str(i+1)+".jpg"))/255 # pixel 0-255 -> 0.0 - 1.0
    imgs.append(img)

test_input = np.array(imgs) # numpy array daraus machen...


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
sd_results = norm[1]

# normalisierung rückgängig machen...
pred_results = pred_results * sd_results + mean_results



# show i. image
i = 10

img = test_input[i]
#cv2.imshow("image", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

plt.imshow(img)

# facial key points
p1 = test_results[i] # Scale up [-1,+1] => [0,90]

plt.scatter(p1[0],p1[1], c="red") # Rechtes Auge
plt.scatter(p1[2],p1[3], c="green") # Linkes Auge
plt.scatter(p1[4],p1[5], c="blue") # Nase
plt.scatter(p1[6],p1[7], c="yellow") # Rechter Mundwinckel
plt.scatter(p1[8],p1[9], c="orange") # Linker Mundwinkel

p1 = pred_results[i] # Scale up [-1,+1] => [0,90]

plt.scatter(p1[0],p1[1], c="red", marker='X') # Rechtes Auge
plt.scatter(p1[2],p1[3], c="green", marker='X') # Linkes Auge
plt.scatter(p1[4],p1[5], c="blue", marker='X') # Nase
plt.scatter(p1[6],p1[7], c="yellow", marker='X') # Rechter Mundwinckel
plt.scatter(p1[8],p1[9], c="orange", marker='X') # Linker Mundwinkel

plt.show()
