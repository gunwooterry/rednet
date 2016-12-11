import numpy as np

X = np.loadtxt("../input/dataX.csv", delimiter=',')
for i in range(X.shape[0]) :
	for j in range(X.shape[1]) :
		if X[(i,j)] != 0 :
			X[(i,j)] = 1+(35+X[(i,j)])/64.0
np.save("../input/dataX_", X)
np.savetxt("../input/dataX_.csv", X, delimiter = ',')
