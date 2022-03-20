import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# read the wine dataset
df = pd.read_csv('dataset/wine.csv')

# use seaborn's pairplot to show correlation among features
sns.pairplot(df)
plt.savefig('pairplot.png')
plt.clf()

# generate pearson correlation matrix
corr_mat = df.corr()

# generate heatmap
sns.set(font_scale=1.2)
sns.heatmap(corr_mat, annot=True, annot_kws={'size':18})
plt.savefig("wine_corr_heat.png")
plt.clf()

# perform feature selection
series = corr_mat['Cultivar']
print(series)

promising = series[(series < -0.5) | (series > 0.5)].index
print(promising)

# create a new dataframe to hold a subset of the original dataframe
candidates_df = pd.DataFrame()
candidates_df[promising] = corr_mat[promising]
candidates_df.drop(columns=['Cultivar'], index=['Cultivar'],
  inplace=True)
print(candidates_df)

skip = []
accept = []
for feature in candidates_df.columns:
  if feature not in skip and feature not in accept:
		# get other features w.r.t. 'feature'
    series = candidates_df[feature]
    print('series =\n', series, '\n', sep='')

		# look for other features that are highly-correlated with 
		# the feature 'colname'
    high_corr = series[(series > 0.6)]
    print('high_corr =\n', high_corr, '\n', sep='')

		# fetch the Pandas series from our candidates
		# only want items that match our 'series' variable
    print(corr_mat['Cultivar'])
    alike = corr_mat['Cultivar'].loc[high_corr.index]
    print('alike =\n', alike, '\n', sep='')

		# idxmax() to get the feature that is most correlated with 
		# "Cultivar" (our target). abs() to absolute the values because 
		# the features could be either positively or negatively 
		# correlated to 'Cultivar'
    top = alike.abs().idxmax()	# row-label (feature-name) of max-value
    accept.append(top)
		
		# do not consider the accepted feature anymore
    alike.drop(top, inplace=True)
    
		# do not consider the remaining features in 'alike' anymore
		# by placing them in the 'skip' basket
    for feature in alike.index:
      if feature not in skip:
        skip.append(feature) 

print('skip = ', skip)
print('selected =', accept)
