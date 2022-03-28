# Workshop - Feature Extraction

</br>

## Objective

</br>

To practice PCA using scikit-learn (sklearn) library

</br>

## Data

</br>

The dataset provided is called ‘wine.csv’ and the description of the data can be found here - [https://archive.ics.uci.edu/ml/datasets/wine](https://archive.ics.uci.edu/ml/datasets/wine)

</br>

## Your Tasks

</br>

- Use Pandas to read in ‘wine.csv’.

</br>

- Use Seaborn's ***kdeplot()*** function to plot the distributions of the following features:
  - Alcalinity of ash
  - Flavanoids
  - Hue
  - Proline

</br>

- The plots would look like this.
  ![distro_no_scaled](https://github.com/cherwah/iss_ml_cert/blob/main/images/distro_no_scaled.png?raw=true)

</br>

- Use SKLearn's **StandardScaler()** to perform feature-scaling on all features, except ***Culitvar***.

  ```python
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt
  from sklearn.decomposition import PCA
  from sklearn.preprocessing import StandardScaler

  # read in dataset
  df = pd.read_csv('dataset/wine.csv')

  # exclude the label 'Cultivar' and convert
  # the dataframe to NumPy array
  data = df.loc[:, "Alcohol":].values

  # perform feature-scaling
  scaled_data = StandardScaler().fit_transform(X=data)
  ```

</br>

- Again, use Seaborn's ***kdeplot()*** function to plot the distributions of the following features:
  - Alcalinity of ash
  - Flavanoids
  - Hue
  - Proline

</br>

- Your plots should now look like this.
  ![distro_scaled](https://github.com/cherwah/iss_ml_cert/blob/main/images/distro_scaled.png?raw=true)

  > Notice the change in values on the x-axes for the features.

</br>

- Select all columns, **except** ***Cultivar***, to perform PCA on.

  > Note that ***Cultivar*** is a label (or a class), hence we do not use it when computing our principal components.

</br>

- Apply PCA on the selected columns with sklearn’s *PCA* function. Feel free to experiment with using different values for **n_components**, as it tells the *PCA* function the number of principal components to generate from our data.

</br>

- The following code applies PCA on dataset 'x' to yield 4 new features (columns of data). The new features are stored in the variable **pc**.

  ```python
  # wants to reduce dimension to only 4
  pca = PCA(n_components=4)

  # applies PCA on dataset 'x' to yield new features
  pc = pca.fit_transform(x) 
  ```

  > Note that The number of rows in each new features is the same as the number of rows in the original dataset. For example, if the original dataset contains 150 rows of data, the new features have 150 rows of new data.

</br>

- Print out the **proportion** of **Explained Variance** that each principal component contributes. Then sum up the ratios and print out Explained Variance for your **n_components** of choice. You can use the member variable **explained_variance_ratio_** from the **pca** object to get the ratios.

</br>

- A sample output is shown below.

  > Explained Variance Ratios = [0.36198848 0.1920749  0.11123631 0.0706903]</br>
  > Total Explained Variance = 0.7359899907589926

</br>

- Determine the minimum number of prinicipal components needed to retain 85% of the original information in the dataset.

</br>

- A sample output is shown below.
  
  > Min principal components = 6

</br>

## References

- [SKLearn website](https://scikit-learn.org/stable)

- [SKLearn's pca() function](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)




