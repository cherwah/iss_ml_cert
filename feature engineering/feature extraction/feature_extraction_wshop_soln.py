import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


'''
Read dataset and return as a Pandas dataset.
Returns None if file is not found.
'''
def read_data(path):
  try:
    return pd.read_csv(path)
  except FileNotFoundError:
    return None


'''
Accepts an array of feature-names to plot on provided dataset
The dataset must be a Pandas DataFrame.

Documentations for a kdeplot can be found here: 
- https://seaborn.pydata.org/generated/seaborn.kdeplot.html
'''
def show_distro(feature_names, data_df, save_to=None, show_win=False):
  num_features = len(feature_names)
  rows = math.ceil(num_features / 2)  # round-up
  cols = 2  # limit to only two columns

  # looks nicer with this style
  sns.set_style(style="darkgrid")

  # create a grid of plots of nrows and ncols
  _, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))

  # plot each given feature-name
  for i in range(rows): # rows
    for j in range(cols):  # columns
      which = i * cols + j  # e.g. 1*2+0 gives us the 3rd elem in the list
      if which < num_features:
        sns.kdeplot(
          data=data_df, 
          x=feature_names[which], 
          hue="Cultivar", # 'hue' allows us to differentiate by type/label
          ax=ax[i, j]
        )  
                  
  # save the plot into an image if save_to parameter is set
  # note that savefig() must come before show() or it would 
  # yield an empty image
  if save_to is not None:
    plt.savefig(save_to)

  # use block=True as parameter so that the plot window remains 
  # on the screen even after the program ends
  if show_win is True:
    plt.show(block=True)  
  

'''
Perform feature scaling / standardization.
Expects provided data to be a NumPy array.
Returns scaled dataset.
'''
def scale_features(data_np):
  return StandardScaler().fit_transform(X=data_np)


'''
Performs PCA on data. The features for the provided dataset 
must already be scaled.
Expects input data to be a NumPy array.
Returns the pca object and the newly-generated features 
'''
def apply_pca(n_components, data_np):
  pca = PCA(n_components=n_components)
  return pca, pca.fit_transform(X=data_np)


'''
Construct a Pandas Dataframe from 'data_np' while using 'ref_data_df'
as a reference (e.g. feature_names and label)
Assumes the label is not found in 'data_np'.
'''
def make_dataframe_from(ref_data_df, data_np, label):   
  data_df = pd.DataFrame(
    data=data_np, # columns of data to fit into reference dataframe
    columns=ref_data_df.columns.drop(label) # want all columns but the label
  )

  # copied the label over to new dataframe
  data_df[label] = ref_data_df[label]
  return data_df


'''
Determines the minimum number of principal components to capture
X% ('percent' parameter) of the original information in our dataset.
'''
def min_components(data_np, percent):
  min = 1

  # shape[1] gives us the number of columns in our dataset
  # +1 to include the last column
  for min in range(1, data_np.shape[1] + 1):
    pca = PCA(n_components=min)
    pca.fit(data_np)  # did not transform (i.e. generate new features)
    if pca.explained_variance_ratio_.sum() >= 0.85:
      return min
    

'''
Entry point for our program
'''
def main():
  # reading in our dataset
  data_df = read_data("dataset/wine.csv")
  if data_df is None:
    print("File is not found.")
    return 

  # plot distributions for these feature-names
  feature_names = [
    "Alcalinity of ash", 
    "Flavanoids",
    "Hue",
    "Proline"
  ]

  # save distro before scaling features  
  show_distro(
    feature_names=feature_names, 
    data_df=data_df, 
    save_to="distro_no_scaled.png"
  )

  # transform dataframe into numpy array.
  # use all columns except 'Cultivar' (our label) since 
  # there is no need to scale our label.
  data_np = data_df.loc[:, "Alcohol":].values

  # scale the features in the numpy array
  scaled_data_np = scale_features(data_np=data_np)

  # convert our scaled numpy data back into a dataframe for plotting distributions
  scaled_data_df = make_dataframe_from(
    ref_data_df=data_df, 
    data_np=scaled_data_np, 
    label='Cultivar'
  )

  # save distro again after scaling features
  show_distro(
    feature_names=feature_names, 
    data_df=scaled_data_df, 
    save_to="distro_scaled.png"
  )
  
  # no need for the new features here, hence the second return
  # value is a '_'
  pca, _ = apply_pca(n_components=4, data_np=scaled_data_np)

  # explained variance ratios capturd by each of the 4 principal components
  print("Explained Variance Ratios =", pca.explained_variance_ratio_)
  
  # total explained variance ratios by all 4 principal components
  print("Total Explained Variance =", pca.explained_variance_ratio_.sum())

  # determine the no. of principal components required to capture
  # at least 85% of the information of the original dataset
  min_pc = min_components(data_np=scaled_data_np, percent=0.85)
  print("Min principal components =", min_pc)


# running via "python feature_extraction_wshop_soln.py"
if __name__ == "__main__":
  main()