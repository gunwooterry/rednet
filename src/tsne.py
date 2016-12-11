import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE


def main():
    raw_data = open("data.csv")
    dataset = np.loadtxt(raw_data, delimiter=',')
    np.random.shuffle(dataset)
    dataset = dataset[:2000, :]
    model = TSNE(n_components=2)
    projection = model.fit_transform(dataset)
    plt.scatter(projection[:, 0], projection[:, 1])
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(projection[:, 0], projection[:, 1], projection[:, 2])
    plt.show()

if __name__ == "__main__":
    main()
