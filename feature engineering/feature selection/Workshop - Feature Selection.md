# Workshop - Feature Selection

## Objective

To practice Feature Selection using Pearson Correlation

</br>

## Data

The dataset provided is called ‘wine.csv’ and the description of the data can be found here - [https://archive.ics.uci.edu/ml/datasets/wine](https://archive.ics.uci.edu/ml/datasets/wine)

</br>

## Your Tasks

- Use Pandas to read in ‘wine.csv’.

  ```python
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt
  
  # read in dataset
  df = pd.read_csv('dataset/wine.csv')
  ```

</br>

- Create a Pair Plot using Seaborn's **pairplot()** function.
  
  ```python  
  # generate pair plot
  sns.pairplot(data_df)
  ```

</br>

- The Pair Plot is shown below.
  ![pairplot](https://github.com/cherwah/iss_ml_cert/blob/main/images/pairplot.png?raw=true)

</br>

- Then using Pandas ***corr()*** function, creates a Pearson Correlation matrix.

  ```python
  # creates a Pearson Correlation matrix from dataframe
  corr_mat = df.corr()	# each value in the matrix is between -1 and 1
  ```

</br>

- Use Seaborn's ***heatmap()*** function to display a Heat map of the Pearson Correlation matrix.

  ```python
  sns.heatmap(corr_mat, # correlation matrix
    annot=True,         # show pearson coefficients
    annot_kws={
      'size':annot_font_size  # font size for pearson coefficients 
    })
  ```

</br>

- The Heat map would look like this.
  ![heatmap](https://github.com/cherwah/iss_ml_cert/blob/main/images/wine_corr_heat.png?raw=true)

</br>

- Perform **Feature Selection** using the Pearson Correlation matrix. 

  1. First, extracting features (candidates) that are either greater than **0.5 negatively-correlated** or greater than **0.5 positively-correlated** to the label, *Cultivar*
  2. Then, for each candidate X, look for peers Y within the rest of the candidates that are **greater than 0.6 positively-correlated** to X
  3. From (X + Y), select the feature that is either the **most negatively or positively correlated** to *Cultivar*
  4. Discard the rest in (X + Y) from the candidates pool
  5. Re-iterate Step 2 until the canddiates pool is empty

</br>

- Print out the **Best Features** from your computation.
  > best features = ['Alcalinity of ash', 'Flavanoids', 'Hue', 'Proline']

</br>

## References

- [Pandas corr() method](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html)

- [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html)
