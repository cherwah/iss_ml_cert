import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


'''
Read dataset and return as a Pandas dataset.
Returns None if file is not found.
'''
def read_data(path):
  try:
    # use Pandas to read in our CSV file
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
Returns a subset of a series by condition.
'''
def features_by_cond(series, less_than, more_than):
  # get back rows of True and False values based on specified condition
  # any rows that met the condition will be True; the rest will be False
  if less_than is not None and more_than is not None:
    # e.g. series[(series < -0.5) | (series > 0.5)]
    series = series[(series < less_than) | (series > more_than)]
  elif less_than is not None:
    # e.g. series[series < -0.5)]
    series = series[(series < less_than)]
  elif more_than is not None:
    # e.g. series[series > 0.5]]
    series = series[(series > more_than)]
  else:
    return None

  return series


'''
Make a new Pandas dataframe from an existing dataframe, based on 
rows of true/false values for conditional selection.
'''
def shrink_corr_mat(corr_mat, keep_feature_names):
  print(keep_feature_names)

  shrinked_mat = corr_mat.loc[keep_feature_names, keep_feature_names]

  print(shrinked_mat)
  return shrinked_mat


'''
Re-order the columns of a dataframe such that the label
occupies the last column and last index in the dataframe.
'''
def move_label_to_last(df, label):
  new_order = list(df.columns)
  new_order.remove(label) # remove the label from the columns
  new_order.append(label) # move label to the last column
  
  # re-index the columns and index within the dataframe
  return df.reindex(columns=new_order, index=new_order)
  

'''
Performs feature-selection. It selects the best features by looking
for features that correlates strongly w.r.t. the label. It also
removes redunduncy by eliminating peers that are strongly-correlated.
'''
def best_features(corr_mat, label, label_limit, peers_limit):
  # only consider features that are high-correlated with our label
  candidates_series = features_by_cond(corr_mat[label], 
    less_than=-label_limit, more_than=label_limit)
  print(candidates_series)

  # shrink original correlation matrix to only consider
  # the features that are highly-correlated with our label
  candidates_df = shrink_corr_mat(corr_mat, 
    keep_feature_names=candidates_series.index)
  print(candidates_df)

  # re-order the columns for easier processing
  candidates_df = move_label_to_last(candidates_df, label)

  # use this to store the best features so far
  accept = [] 

  # iteratively compares features against peers and the label
  while len(candidates_df) > 1:  # stop when left with our label
    # inspect each feature in turn
    feature = candidates_df.columns[0]

    # get all peers of these feature, except the label
    series = candidates_df.loc[feature].drop(label)
    print('series =\n', series, '\n', sep='')

    # look for other features that are highly-correlated with 
    # the current 'feature'. only consider positively correlated 
    # to remove redundancy.
    high_corr = features_by_cond(series, 
      less_than=None, more_than=peers_limit)
    print('high_corr =\n', high_corr, '\n', sep='')

    # extract the pearson correlation values of each of these 
    # highly-correlated features w.r.t. our label
    alike = candidates_df.loc[high_corr.index, label]
    print('alike =\n', alike, '\n', sep='')

    # idxmax() to get the feature that is most correlated with 
    # our label. abs() to absolute the values because 
    # the features could be either positively or negatively 
    # correlated to our label
    top = alike.abs().idxmax()  # row-label (feature-name) of max-value
    accept.append(top)

    # place index-names into an array, so that we can use them
    # as parameters to the drop() function
    alike = list(alike.index) 

    # done with feature, remove feature from 'candidates_df',
    # this allows our candidates to be smaller each time
    candidates_df.drop(columns=alike, index=alike, inplace=True)

  # returns best features in the    
  return accept

  
'''
Entry point for our program
'''
def main():
  # reading in our dataset
  df = read_data('dataset/wine.csv')
  if df is None:
    print('File is not found.')
    return

  # create correlation matrix
  corr_mat = df.corr()

  # generate plots
  make_pairplot(data_df=df, save_to='pairplot.png')
  make_heatmap(corr_mat=corr_mat, save_to='wine_corr_heat.png')

  # feature selection - find the best features that correlates 
  # to our label; use these features to train our model for
  # future predictions
  features = best_features(
    corr_mat=corr_mat, 
    label='Cultivar',
    label_limit=0.5, 
    peers_limit=0.6)

  print('best features =', features)


# running via "python feature_selection_wshop_soln.py"
if __name__ == '__main__':
  main()