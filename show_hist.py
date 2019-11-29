#!/usr/local/bin/python3
import matplotlib.pyplot as plt
import numpy as np

# laden trainingsverlauf (loss,val_loss)
hist = np.load('hist1.npy',allow_pickle=True)
hist = hist.item()

### zeige trainingsverlauf als grafik
for label in hist.keys():
    plt.plot(hist[label],label=label) # alle werte plotten
plt.xlabel("epochs")
plt.legend()
plt.title("The final val_loss={:4.3f}".format(hist["val_loss"][-1])) # letzter wert im array = [-1]
plt.show()
