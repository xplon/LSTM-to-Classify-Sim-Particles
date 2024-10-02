import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

path = './data/train/train_balanced.pickle'
df = pd.read_pickle(path)

# train_df, df = train_test_split(df, test_size=0.001, random_state=1)

labels = np.array(df['Type'])  # 示例标签，0和1代表不同的分类
# mask = np.array(df['Mask'])

# 将Event_Vector转换为NumPy数组
X = np.array(df['Event_Vector'].tolist())

print(X.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking

# 构建LSTM模型
model = Sequential()

# 如果需要忽略填充的 (0, 0) 对，应添加 Masking 层
model.add(Masking(mask_value=0, input_shape=(21, 2)))

# 添加LSTM层
model.add(LSTM(50, return_sequences=False))  # 50是LSTM单元的数量

# 添加输出层，根据任务选择激活函数
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 设置模型保存的文件路径和名称
checkpoint_path = "./models/model_epoch_{epoch:02d}.keras"

# 创建ModelCheckpoint回调
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    save_best_only=False,
    monitor='loss',
    verbose=1
)

# 训练模型，并将ModelCheckpoint回调传入
model.fit(
    X, labels,
    epochs=10,
    batch_size=32,
    callbacks=[checkpoint]  # 将checkpoint回调传入fit函数
)

