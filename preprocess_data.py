import pandas as pd


df = pd.read_csv('shrub_dataset.csv')

df
"""
     Leave size (cm)  Shrub height (m)     Shrub species name
0           8.232398          3.064781            Hazel Shrub
1           6.374936          1.973804            Hazel Shrub
2           8.961280          3.854265            Hazel Shrub
3           8.242065          2.412739            Hazel Shrub
4           6.736104          2.559504            Hazel Shrub
...
...
97         4.047278          1.403136  Alder Buckthorn Shrub
98         5.911174          2.655614  Alder Buckthorn Shrub
99         4.131060          1.906048  Alder Buckthorn Shrub
"""

df.count()
"""
Leave size (cm)       100
Shrub height (m)      100
Shrub species name    100
"""

df['Shrub species name'].unique()

"""
array(['Hazel Shrub', 'Alder Buckthorn Shrub'],
      dtype=object)
"""

#########################################################

import pandas as pd

df = pd.read_csv('shrub_dataset.csv')
# Extract the class column from the dataframe
# and conver the class names to numbers.
df_values = df.values
class_column = df_values[:, 2:3]
class_column[class_column == 'Hazel Shrub'] = 0
class_column[class_column == 'Alder Buckthorn Shrub'] = 1

# Drop the class column in the original df
df2 = df.drop(columns=['Shrub species name'])
# Normalise the features in the df
preprocessed_df=(df2-df2.min())/(df2.max()-df2.min())

# Insert the class column again
preprocessed_df.insert(2, 'Shrub species name', class_column)

# Write the preprocessed df to a csv
preprocessed_df.to_csv('preprocessed_shrub_dataset.csv')
