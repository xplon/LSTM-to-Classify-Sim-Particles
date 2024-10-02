import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split


path = './data/test/test_3.parquet'
df = pd.read_parquet(path)

print(df.head(50))
# train_df, df = train_test_split(df, test_size=0.00001, random_state=1)

num_doms = 21
dom_time_cols = [f'DomId_{i}' for i in range(num_doms)]
dom_intensity_cols = [f'NDomId_{i}' for i in range(num_doms)]

def create_event_vector_with_mask(row):
    # 创建一个mask来记录时间和强度向量中的哪些值是有效的
    mask = np.array([1 if pd.notna(row[time_col]) else 0 for time_col in dom_time_cols])
    # 将时间和强度向量以及mask合并
    return mask

# 将每行转换为一个目标向量
df['Mask'] = df.apply(create_event_vector_with_mask, axis=1)

def create_event_vector(row):
    # 创建包含 (DomId, NDomId) 对的列表
    vector = np.array([(row[time_col] if pd.notna(row[time_col]) else 0,
               row[intensity_col])
              for time_col, intensity_col in zip(dom_time_cols, dom_intensity_cols)])
    return vector

# 将每行转换为一个包含 (DomId, NDomId) 对的列表
df['Event_Vector'] = df.apply(create_event_vector, axis=1)

newdf = df[['eventID', 'Event_Vector', 'Type']]

# newdf = newdf[newdf['Type'] != 1]
# a = np.array(np.array(df['Event_Vector']))
# print((np.stack(df['Event_Vector'].values)).shape)

# newdf.to_parquet('./data/test/test_5.parquet')
newdf.to_pickle('./data/test/test.pickle')


# # 假设您有标签数据
# labels = np.array([0, 1])  # 示例标签，0和1代表不同的分类
#
# # 将Event_Vector转换为NumPy数组
# X = np.array(df['Event_Vector'].tolist())
#
# # 检查X的形状，确保它可以用于LSTM模型
# print(X)  # 输出应该是 (样本数量, 时间步长, 特征维度)
# print(newdf['Event_Vector'])
