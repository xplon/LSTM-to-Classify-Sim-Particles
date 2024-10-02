#%%
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# 1. 加载测试数据
path = './data/test/test-type0.pickle'
df = pd.read_pickle(path)

#%%
X_test = np.array(df['Event_Vector'].tolist())
# Y_test = np.array(df['Type'])

# # 2. 加载保存的模型
model = load_model("./models/model_epoch_10.keras")

# 3. 对测试数据进行预测
predictions = model.predict(X_test)

# 4. 处理预测结果（如果是二分类任务）
predicted_labels = (predictions > 0.5).astype(int)

# # 5. 保存或展示预测结果
# with open('predictions.pickle', 'wb') as f:
#     pickle.dump(predicted_labels, f)

# 1. 加载测试数据
path = './data/test/test-type1.pickle'
df = pd.read_pickle(path)

X_test1 = np.array(df['Event_Vector'].tolist())
# Y_test = np.array(df['Type'])

# 3. 对测试数据进行预测
predictions1 = model.predict(X_test1)

# 4. 处理预测结果（如果是二分类任务）
predicted_labels1 = (predictions1 > 0.5).astype(int)

# # 5. 保存或展示预测结果
# with open('predictions.pickle', 'wb') as f:
#     pickle.dump(predicted_labels, f)

#%%
np.count_nonzero(predicted_labels1)

#%%
import numpy as np
import matplotlib.pyplot as plt

test_predictions = predictions.flatten()
data = test_predictions
x=np.array(data)

original_counts, bins, patches = plt.hist(x, bins=20, density=False)
plt.cla()
# counts, bins, patches = plt.hist(x, bins=20, density=True, alpha=0.6, color='g')

total_counts = sum(original_counts)
fractions = original_counts / total_counts

heights = [patch.get_height() for patch in patches]
heights = heights / total_counts

plt.bar(bins[:-1], fractions, width=np.diff(bins), alpha=0.5, color='orange')

test_predictions1 = predictions1.flatten()
data1 = test_predictions1
x1=np.array(data1)

original_counts1, bins1, patches1 = plt.hist(x1, bins=20, density=False, alpha=0)


total_counts1 = sum(original_counts1)
fractions1 = original_counts1 / total_counts1

heights1 = [patch1.get_height() for patch1 in patches1]
heights1 = heights1 / total_counts1

plt.bar(bins1[:-1], fractions1, width=np.diff(bins1), alpha=0.5, color='blue')

# plt.xscale('log')
plt.yscale('log')
plt.ylim([0, 1])
# sns.kdeplot(x, color="blue")

plt.xlabel('type')
plt.ylabel('Density')

# Annotate bars with the fraction of each bin
for fraction, h, patch in zip(fractions, heights, patches):
    height = 0
    if h != 0:
        height = h
    if height > 0:  # Only annotate non-zero bars
        plt.annotate(f'{fraction:.2%}',  # Display as a percentage
                    xy=(patch.get_x() + patch.get_width() / 2.0, height),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom')

for fraction, h, patch in zip(fractions1, heights1, patches1):
    height = 0
    if h != 0:
        height = h
    if height > 0:  # Only annotate non-zero bars
        plt.annotate(f'{fraction:.2%}',  # Display as a percentage
                    xy=(patch.get_x() + patch.get_width() / 2.0, height),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom')

plt.show()
plt.savefig('./pic/p4.png')
