import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
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
accepts an array of features to plot
'''
def plot_distributions(feature_names):
  pass


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
entry point for our program
'''
def main():
  # begin by reading in our dataset
  data = read_data("../../wine.csv")
  if data == None:
    print("File is not found.")
    return 


  pass


'''
running via python feature_extraction_wshop_soln.py
'''
if __name__ == "__main__":
  main()