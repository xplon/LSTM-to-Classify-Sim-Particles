import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# 1. 加载测试数据
path = './data/test/test.pickle'
df = pd.read_pickle(path)

X_test = np.array(df['Event_Vector'].tolist())
Y_test = df['Type']

# 2. 加载保存的模型
model = load_model("./models/model_epoch_10.keras")

# 3. 对测试数据进行预测
test_predictions = model.predict(X_test)

# 1. 加载测试数据
path = './data/train/train.pickle'
df = pd.read_pickle(path)

train_X = np.array(df['Event_Vector'].tolist())
train_y = df.Type

# 3. 对测试数据进行预测
train_y_pre = model.predict(train_X).flatten()

import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error

path = './data/test/test.pickle'
df = pd.read_pickle(path)
df.reset_index(inplace=True)
# val_y = np.array(df.Type)
val_y = df.Type
test_predictions = test_predictions.flatten()

variance = np.var(val_y - test_predictions)
print("方差：", variance)
# print("MAE：", mean_absolute_error(val_y, test_predictions))

# li_model = LinearRegression()
# X = np.array(val_y).reshape(-1, 1)  # -1表示自动计算维度，因此效果等同于`.reshape(5, 1)`
# li_model.fit(X, test_predictions)  # 训练模型
#
# X_v = np.arange(80, 160).reshape(-1, 1)
# y_v = li_model.predict(X_v)
#
# plt.plot(X_v, y_v, c='blue')
# plt.show()
# print("截距：", li_model.intercept_)
# print("斜率：", li_model.coef_)

# # Plot a scatter plot of predicted vs actual values for the test set
# plt.figure(figsize=(10, 10))
# # plt.scatter(val_y, test_predictions, alpha=0.5)
# plt.scatter(Y_test, test_predictions, alpha=0.01)
# plt.plot([0, val_y.max()], [0, val_y.max()], 'r--')
# plt.xlabel('Actual Type')
# plt.ylabel('Predicted Type')
# plt.title('Actual vs Predicted Type')
# # plt.show()
# plt.savefig('./pic/p1.png')
#
# # Plot a histogram of the residuals (differences between actual and predicted values) and fit a Gaussian
# train_residuals = train_y - train_y_pre
# test_residuals = val_y - test_predictions
#
# mae = mean_absolute_error(val_y, test_predictions)
# print(f"MAE Score: {mae}")
#
# # Fit Gaussian to the residuals
# mu_train, std_train = norm.fit(train_residuals)
# mu_test, std_test = norm.fit(test_residuals)
#
# # Calculate FWHM
# fwhm_train = 2 * np.sqrt(2 * np.log(2)) * std_train
# print(std_train, " ", std_test)
# fwhm_test = 2 * np.sqrt(2 * np.log(2)) * std_test
#
# plt.figure(figsize=(10, 6))
# # Plot training residuals
# plt.hist(train_residuals, bins=500, alpha=0.5, density=True, label='Train Residuals')
# # plt.xlim(-0.5, 0.5)
# x_train = np.linspace(train_residuals.min(), train_residuals.max(), 100)
# p_train = norm.pdf(x_train, mu_train, std_train)
# plt.plot(x_train, p_train, 'b', linewidth=2, label=rf'Train Gaussian fit: $\mu={mu_train:.2f}$, $\sigma={std_train:.2f}$')
# plt.axvline(mu_train - fwhm_train / 2, color='blue', linestyle='dashed', linewidth=1)
# plt.axvline(mu_train + fwhm_train / 2, color='blue', linestyle='dashed', linewidth=1)
# plt.text(mu_train, max(p_train)*0.8, f'Train FWHM = {fwhm_train:.2f}', color='blue', ha='center')
#
# # Plot testing residuals
# plt.hist(test_residuals, bins=500, alpha=0.5, density=True, label='Test Residuals')
# # plt.xlim(-0.5, 0.5)
# x_test = np.linspace(test_residuals.min(), test_residuals.max(), 100)
# p_test = norm.pdf(x_test, mu_test, std_test)
# plt.plot(x_test, p_test, 'r', linewidth=2, label=rf'Test Gaussian fit: $\mu={mu_test:.2f}$, $\sigma={std_test:.2f}$')
# plt.axvline(mu_test - fwhm_test / 2, color='red', linestyle='dashed', linewidth=1)
# plt.axvline(mu_test + fwhm_test / 2, color='red', linestyle='dashed', linewidth=1)
# plt.text(mu_test, max(p_test)*0.9, f'Test FWHM = {fwhm_test:.2f}', color='red', ha='center')
#
#
# plt.xlabel('Residuals (Degrees)')
# plt.ylabel('Density')
# plt.title('Distribution of Residuals with Gaussian Fit')
# plt.legend()
# # plt.show()
# plt.savefig('./pic/p2.png')


# from scipy.stats import gaussian_kde
# from matplotlib.colors import Normalize
#
# # 使用高斯核密度估计计算密度
# xy = np.vstack([test_predictions, val_y])
# kde = gaussian_kde(xy)
# z = kde(xy)
#
# # 创建一个新的图形
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 创建网格点
# x = np.linspace(140, max(test_predictions), 100)
# y = np.linspace(140, max(val_y), 100)
# X, Y = np.meshgrid(x, y)
# positions = np.vstack([X.ravel(), Y.ravel()])
# Z = np.reshape(kde(positions).T, X.shape)
#
# # 绘制三维曲面图
# ax.plot_surface(X, Y, Z, cmap='viridis')
#
# # 设置轴标签
# ax.set_xlabel('Test Predictions')
# ax.set_ylabel('Val Y')
# ax.set_zlabel('Density')
#
# mappable = plt.cm.ScalarMappable(cmap='viridis')
# mappable.set_array(Z)
# fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
#
# ax.view_init(elev=90, azim=270)
# 显示图形

plt.hist2d(test_predictions, val_y, bins=30, cmap='Blues')

# 添加颜色栏以表示频率
plt.colorbar(label='Frequency')

# 添加标题和标签
plt.title('2D Histogram')
plt.xlabel('Prediction Type')
plt.ylabel('Actual Type')
plt.show()
plt.savefig('./pic/p3.png')