from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('o','s','^', 'v','<')
    colors = ('red', 'blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])