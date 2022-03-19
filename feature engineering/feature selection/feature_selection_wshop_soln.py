import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
Generate Pair-Plot from a Pandas dataframe and
save it to a file. 
'''
def make_pairplot(data_df, save_to):
  sns.pairplot(data_df) # generate pair-plot
  plt.savefig(save_to)  # save to file
  plt.clf() # clear plot; next plot starts afresh


'''
Generate a Heat Map from a correlational matrix and
save it to a file.
'''
def make_heatmap(corr_mat, save_to, font_scale=1.2, annot_font_size=18):
  sns.set(font_scale=font_scale)

  # generate heat-map
  sns.heatmap(corr_mat, # correlation matrix
    annot=True,         # pearson correlation values
    annot_kws={
      'size':annot_font_size  # font size for values
    })

  plt.savefig("wine_corr_heat.png") # save to file
  plt.clf()


'''
Get feature-names based on input series 'bool_rows'
that contains rows of true/false values.
'''
def get_features(corr_mat, target, less_than, more_than):
  series = corr_mat[target]

  # get back rows of True and False values based on specified condition
  # any rows that met the condition will be True; the rest will be False
  if less_than is not None and more_than is not None:
    # e.g. series[(series < -0.5) | (series > 0.5)]
    bool_rows = series[(series < less_than) | (series > more_than)]
  elif less_than is not None:
    # e.g. series[series < -0.5)]
    bool_rows = series[(series < less_than)]
  elif more_than is not None:
    # e.g. series[series > 0.5]]
    bool_rows = series[(series > more_than)]
  else:
    return None

  series = series[bool_rows]  # Pandas series
  return series.index         # get feature-names


'''
Make a new Pandas dataframe from an existing dataframe, based on 
rows of true/false values for conditional selection.
'''
def make_dataframe_from(corr_mat, target, bool_rows, rm_target=False):
  series = corr_mat[target]      # get column that corresponds to the target  
  series = series[bool_rows]    # extract values of corresponding True rows
  feature_names = series.index  # get the index-values of the Pandas series

  df = pd.DataFrame() # create empty Pandas dataframe
  df[feature_names] = corr_mat[feature_names] # copy values over

  if rm_target is True:
    df.drop(columns=[target], index=[target], inplace=True)

  return df


def best_features(corr_mat, label, label_limit, peers_limit):
  skip, accept = [], []
  
  cond = 
  candidates_df = make_dataframe_from(corr_mat=corr_mat,
    target=label, ))
  for feature 



  
'''
Entry point for our program
'''
def main():
  # reading in our dataset
  df = read_data("dataset/wine.csv")
  if df is None:
    print("File is not found.")
    return

  # create correlation matrix
  corr_mat = df.corr()

  # generate plots
  make_pairplot(data_df=df, save_to="pairplot.png")
  make_heatmap(corr_mat=corr_mat, save_to="wine_corr_heat.png")

  features = best_features(corr_mat=corr_mat, label='Cultivar',
    label_limit=0.5, peers_limit=0.6)
  print("best features =", features)



# running via "python feature_selection_wshop_soln.py"
if __name__ == "__main__":
  main()