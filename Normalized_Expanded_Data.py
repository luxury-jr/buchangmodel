import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 加载原始数据
data = pd.read_excel('Transposed_宇海数据.xlsx')

# 确保原始数据只有 25 组
original_data = data.head(25)

# 执行 Bootstrap 抽样以扩充到 125 组数据
bootstrap_samples = original_data.sample(n=125, replace=True, random_state=42)
print(bootstrap_samples)

# 初始化 MinMaxScaler 用于数据规范化
scaler = MinMaxScaler()

# 拟合并转换扩充后的数据
normalized_data = scaler.fit_transform(bootstrap_samples)
# print(normalized_data)

# 将归一化后的数据转换回 DataFrame
normalized_df = pd.DataFrame(normalized_data, columns=bootstrap_samples.columns)

# 将归一化后的数据保存到 Excel 文件
normalized_df.to_excel('Normalized_Expanded_Data.xlsx', index=False)

print('扩充并归一化后的数据已保存至 "Normalized_Expanded_Data.xlsx"')
