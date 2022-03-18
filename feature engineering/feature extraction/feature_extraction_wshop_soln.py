import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import math
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


'''
read dataset and return as a pandas dataset
returns None if file is not found
'''
def read_data(path):
  try:
    return pd.read_csv(path)
  except FileNotFoundError:
    return None


'''
accepts an array of feature-names to plot on provided data
'''
def show_distro(feature_names, data, save_to=None):
  num_features = len(feature_names)
  rows = math.ceil(num_features / 2)  # round-up
  cols = 2  # limit to only two columns

  # looks nicer with this style
  sb.set(style='darkgrid')  

  # create a grid of plots of nrows and ncols
  _, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))

  # plot each given feature-name
  for i in range(rows): # rows
    for j in range(cols):  # columns
      which = i * cols + j  # e.g. 1*2+0 gives us the 3rd elem in the list
      if which < num_features:
        sb.kdeplot(data=data, x=feature_names[which],
          hue='Cultivar', ax=ax[i, j])

  # save the plot into an image if save_to parameter is set
  # note that savefig() must come before show() or it would 
  # yield an empty image
  if save_to is not None:
    plt.savefig(save_to)

  # use block=True as parameter so that the plot window remains 
  # on the screen even after the program ends
  plt.show(block=True)  
  

'''
Perform feature scaling / standardization.
Returns scaled dataset.
'''
def scale_features(data):
  pass


'''
Performs principal component analysis on data. The features for the 
provided dataset must already be scaled.
Returns new features after applying PCA.
'''
def apply_pca(n_pca_components, data):
  pass


'''
Determines the minimum number of principal components to catpure
85% of the original information in our dataset.
'''
def min_components(data):
  pass


'''
entry point for our program
'''
def main():
  # begin by reading in our dataset
  data = read_data("dataset/wine.csv")
  if data is None:
    print("File is not found.")
    return 

  # plot distributions for these feature-names
  feature_names = [
    'Alcalinity of ash', 
    'Flavanoids',
    'Hue',
    'Proline'
  ]

  # save distro before scaling features
  show_distro(feature_names, data, "distro_no_scaled.png")

  # scale the features of our dataset


  pass


'''
running via python feature_extraction_wshop_soln.py
'''
if __name__ == "__main__":
  main()