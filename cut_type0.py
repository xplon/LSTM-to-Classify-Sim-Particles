import pandas as pd

path = './data/train/train.pickle'
df = pd.read_pickle(path)

type0 = df[df['Type'] == 0]
type1 = df[df['Type'] == 1]

# 抽样 type0 的数据
type0_sampled = type0.sample(n=len(type1), random_state=42)

# 合并并重新打乱数据
balanced_df = pd.concat([type0_sampled, type1]).sample(frac=1, random_state=42)

balanced_df.to_pickle('./data/train/train_balanced.pickle')