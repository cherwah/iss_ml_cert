import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


data = np.loadtxt('../data/iris.csv', delimiter=',', 
	dtype='str', skiprows=1)
print(data)

# get all rows; exclude first and last column
x = data[:,1:-1]
print(x)

# get labels
y = data[:,-1]
print(y)

# use StandardScaler to zero-mean and unit-variant 
# on each extracted column (note that each column 
# represents multiple observations of a single feature 
# in our dataset)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# pass standardized data to PCA
# out of the 4 existing features,
# we want to create 2 NEW features
pca = PCA(n_components=2)

# returns a NumPy array of size 'n_components'
pc = pca.fit_transform(x_scaled)

for species in np.unique(y):
	pc_by_species = pc[y==species]
	plt.scatter(x=pc_by_species[:, 0], 
		y=pc_by_species[:, 1],
        label=species)

plt.legend()
plt.title('After PCA Transformation')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.sum())