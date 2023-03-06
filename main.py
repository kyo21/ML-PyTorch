from data.dataloader import Dataloader
from perceptron import Perceptron
import matplotlib.pyplot as plt

dataload = Dataloader()

X, y = dataload.read_data()

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_)+1),
         ppn.errors_, marker='o')

plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

