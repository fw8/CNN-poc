#!/usr/local/bin/python3
import matplotlib.pyplot as plt
import yaml

# laden trainingsverlauf (loss,val_loss)
with open("hist1.yaml", "r") as yaml_file:
    hist = yaml.load(yaml_file)

### zeige trainingsverlauf verlauf als grafik
for label in hist.keys():
    plt.plot(hist[label],label=label) # alle (beide) werte plotten
plt.xlabel("epochs")
plt.legend()
plt.title("The final val_loss={:4.3f}".format(hist["val_loss"][-1])) # letzter wert im array = [-1]
plt.show()
