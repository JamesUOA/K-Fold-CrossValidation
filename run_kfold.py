from KfoldBERT import KfoldBERT
import pandas as pd

df = pd.read_csv("train.csv", encoding = "ISO-8859-1")
sentences = list(df.sentence.values)
labels = list(df.label.values)

kfold = KfoldBERT(sentences,labels, 3)
kfold.run()

print("Accuracy: ", kfold.getAccuracy())
print("Loss: ", kfold.getLoss())