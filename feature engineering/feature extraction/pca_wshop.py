import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('../data/wine.csv')
df = pd.DataFrame(dataset)
pd.set_option('display.max_columns', None)
print(df.head(n=30))

y = dataset.loc[:,'Cultivar'].values
x = StandardScaler().fit_transform(dataset.iloc[:,1:])

# apply PCA on our features
pca = PCA(n_components=6)
pc = pca.fit_transform(x)
print(pc)

print('Explained Variance ratio: ', pca.explained_variance_ratio_)
print('Explained Variance for ' + str(pca.n_components) + 
	' principal components: ', pca.explained_variance_ratio_.sum())

fig = plt.figure(figsize=(5, 5))
ax = plt.axes(projection='3d')

for i in np.unique(y):
	pc_by_cultivar = pc[y==i]
	ax.scatter3D(xs=pc_by_cultivar[:, 0], 
		ys=pc_by_cultivar[:, 1],
		zs=pc_by_cultivar[:, 2], 
		label='Cultivar ' + str(i))

plt.title('PCA on wine dataset')
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_zlabel("Principal Component 3")
plt.legend()
plt.show()
