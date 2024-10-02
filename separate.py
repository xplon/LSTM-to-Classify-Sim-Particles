# this file is used to separate all the data into train data and test data randomly.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_parquet('./data/train/full-fixed_3.parquet')

train_df, test_df = train_test_split(df, test_size=0.2, random_state=1)

train_df.to_parquet('./data/train/train_3.parquet')
test_df.to_parquet('./data/test/test_3.parquet')
