import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
def get_features(corr_mat, label, bool_rows):
  series = corr_mat[label]
  series = series[bool_rows]  # Pandas series
  return series.index   # get feature-names


'''

'''
def make_dataframe_from(corr_mat, label, bool_rows, no_label=True):
  series = corr_mat[label]      # get column that corresponds to the label  
  series = series[bool_rows]    # extract values of corresponding True rows
  feature_names = series.index  # get the index-values of the Pandas series

  df = pd.DataFrame() # create empty Pandas dataframe
  df[feature_names] = corr_mat[feature_names] # copy values over

  if no_label is True:
    df.drop(columns=[label], index=[label], inplace=True)

  return df

  





'''
Entry point for our program
'''
def main():
  pass


# running via "python feature_selection_wshop_soln.py"
if __name__ == "__main__":
  main()