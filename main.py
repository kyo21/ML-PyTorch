from data.dataloader import Dataloader
from perceptron import Perceptron
from ADALine import AdalineSGD
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('o','s','^', 'v','<')
    colors = ('red', 'blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min( - 1, X[:,1]).max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    

dataload = Dataloader()

X, y = dataload.read_data()
X_stad = dataload.standardization(X)

#---------------------Perceptron--------------------------#

# ppn = Perceptron(eta=0.1, n_iter=10)
# ppn.fit(X, y)

# plt.plot(range(1, len(ppn.errors_)+1),
#          ppn.errors_, marker='o')

# plt.xlabel('Epochs')
# plt.ylabel('Number of updates')
# plt.show()

#----------------Adaline-------------------------#
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
ada1 = AdalineSGD(n_iter=20, eta=0.005).fit(X,y)
ax[0].plot(range(1, len(ada1.losses_)+1),
           np.log10(ada1.losses_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Mean Squared Error)')
ax[0].set_title('Adaline with leanring rate 0.005')

ada2 = AdalineSGD(n_iter=20, eta=0.005).fit(X_stad,y)
ax[1].plot(range(1, len(ada2.losses_)+1),
           np.log10(ada2.losses_), marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Mean Squared Error)')
ax[1].set_title('Adaline with learning rate 0.005 and X_stad')
plt.show()