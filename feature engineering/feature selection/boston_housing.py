import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

# for documentation of load_bostom(), refer to
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html 
# and https://h1ros.github.io/posts/loading-scikit-learns-boston-housing-dataset
boston = load_boston()

# setting up our dataset to perform pearson correlation
# boston.feature_names does not include 'y' or the 'label'
# our target is MEDV, which is the "median value in $1000's".
target = 'MEDV'
columns = np.append(boston.feature_names, target)

# create a Pandas dataframe, then produce a correlation-matrix 
# out of the dataframe.
df = pd.DataFrame(
	np.append(boston.data,
	# appending the label (i.e. boston.target) to the features (boston.data)
	np.reshape(boston.target, (-1,1)), axis=1),
	columns=columns) 
print(df)

# create the correlational matrix
corr_mat = df.corr()
print(corr_mat)

plt.figure(figsize=(13,10))
sns.heatmap(data=corr_mat, annot=True, cmap='GnBu')
plt.title('Correlation Matrix for Boston Housing dataset')
plt.show()

target_column = corr_mat[target]
target_column.drop([target], inplace=True)
candidates = target_column[target_column.abs() > 0.5]
print('candidates w.r.t. MEDV =\n', candidates, '\n', sep='')

to_drop = list(set(corr_mat.index) - set(candidates.index))
print('drop the following from corr_mat =', to_drop, '\n')

workset = corr_mat.drop(index=to_drop, columns=to_drop)
print('after dropping =\n', workset, '\n', sep='')

skip = []
accept = []
for colname in workset.columns:
	if colname not in skip and colname not in accept:
		# get all features w.r.t. 'colname'
		series = workset[colname]
		print('series =\n', series, '\n', sep='')

		# look for other features that are highly-correlated with 
		# the feature 'colname'
		series = series[(series > 0.6)]
		print('series =\n', series, '\n', sep='')

		# fetch the Pandas series from the 'candidates' dataframe. 
		# only want items that match our 'series' variable
		alike = candidates[series.index]
		print('alike =\n', alike, '\n', sep='')

		# idxmax() to get the feature that is most correlated with 
		# "MEDV" (our target). abs() to absolute the values because 
		# the features could be either positively or negatively 
		# correlated to 'MEDV'
		top = alike.abs().idxmax()	# row-label (feature-name) of max-value
		accept.append(top)
		
		# do not consider the accepted feature anymore
		alike.drop(top, inplace=True)

		print('index = ', alike.index, '\n')

		# do not consider the remaining features in 'alike' anymore
		# by placing them in the 'skip' basket
		for feature in alike.index:
			if feature not in skip:
				skip.append(feature) 
		# skip += set(alike.index) - set([top])

print('skip = ', skip)
print('selected =', accept)

