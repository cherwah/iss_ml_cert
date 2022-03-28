# Workshop - Feature Selection



**Objective:** 

To practice Feature Selection using Pearson Correlation



**Data**

The dataset provided is called ‘wine.csv’ and the description of the data can be found here - https://archive.ics.uci.edu/ml/datasets/wine

 

**Your Tasks**

- Use Pandas to read in ‘wine.csv’.

  

- Create a Pair Plot using Seaborn's ***pairplot***() function. 

  ```python
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt
  
  # read in dataset
  df = pd.read_csv('dataset/wine.csv')
  
  # generate pair plot
  sns.pairplot(data_df) # generate pair-plot
  ```

  

- The Pair Plot is shown below.

  ![](/Users/cherwah/git_repos/iss_ml_cert/pairplot.png)

  

- Then using Pandas ***corr***() function, creates a Pearson Correlation matrix.

  ```python
  # creates a Pearson Correlation matrix from dataframe
  corr_mat = df.corr()	# each value in the matrix is between -1 and 1
  ```

  

- Use Seaborn's ***heatmap***() function to display a Heat map of the Pearson Correlation matrix.

  ```python
  sns.heatmap(corr_mat, # correlation matrix
  	annot=True,         # show pearson coefficients
    annot_kws={
    	'size':annot_font_size  # font size for pearson coefficients 
    })
  ```

  

- The Heat map would look like this.

  ![image-20220328134938108](/Users/cherwah/Library/Application Support/typora-user-images/image-20220328134938108.png)

  

- Perform **Feature Selection** using the Pearson Correlation matrix. 

  1. First, extracting features (candidates) that are either greater than **0.5 negatively-correlated** or greater than **0.5 positively-correlated** to the label, *Cultivar*
  2. Then, for each candidate X, look for peers Y within the rest of the candidates that are **greater than 0.6 positively-correlated** to X
  3. From (X + Y), select the feature that is either the **most negatively or positively correlated** to *Cultivar*
  4. Discard the rest in (X + Y) from the candidates pool
  5. Re-iterate Step 2 until the canddiates pool is empty

  

- Print out the **Best Features** from your computation.

  > best features = ['Alcalinity of ash', 'Flavanoids', 'Hue', 'Proline']

  

**References**

- Pandas' **corr()** method - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html

- Pandas **DataFrame** - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html

  