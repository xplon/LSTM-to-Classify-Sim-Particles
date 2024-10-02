import numpy as np
import pandas as pd

ang_file_path = './data/train/full.parquet'
ang_data = pd.read_parquet(ang_file_path)

# ang_data.replace(np.nan, -10000, inplace=True)

columns_to_check = [f'DomId_{i}' for i in range(21)]
ang_data[columns_to_check] = ang_data[columns_to_check].replace(0, np.nan)


# Define a function to change arrival time into relative time (time - the time this particle (event) first detected)
def subtract_min(row):
    min_value = row[2:-23].min(skipna=True)
    row[2:-23] = row[2:-23].apply(lambda x: x - min_value if pd.notna(x) else x)
    return row


# apply to every column in the file
ang_data = ang_data.apply(subtract_min, axis=1)

print(ang_data.head())

# This file will be processed into full-fixed_3.parquet
ang_data.to_parquet('./data/train/full-fixed_2.parquet')
