import pandas as pd
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 加载数据
data_transposed = pd.read_excel('Transposed_扩充数据_Bootstrap.xlsx')

# 确保数据按时间顺序排列（如果有时间戳）

# 划分训练集和测试集（假设您已经按时间顺序排列了数据）
n_obs = int(len(data_transposed) * 0.3)  # 30% 作为测试集
train, test = data_transposed[0:-n_obs], data_transposed[-n_obs:]

# 创建 VAR 模型
model = VAR(train)
model_fitted = model.fit()

# 预测
lag_order = model_fitted.k_ar
forecast_input = train.values[-lag_order:]  # 获取训练数据的最后几个观察值
forecast = model_fitted.forecast(y=forecast_input, steps=n_obs)

# 将预测结果转换为 DataFrame
forecast_df = pd.DataFrame(forecast, index=test.index, columns=test.columns)

# 计算评估指标
mae = mean_absolute_error(test, forecast_df)
mse = mean_squared_error(test, forecast_df)
r2 = r2_score(test, forecast_df)

print("VAR Model - MAE:", mae, "MSE:", mse, "R2:", r2)
