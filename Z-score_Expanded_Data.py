import pandas as pd
from sklearn.preprocessing import StandardScaler

# 加载原始数据
data = pd.read_excel('Transposed_宇海数据.xlsx')

# 确保原始数据只有 25 组
original_data = data.head(25)

# 执行 Bootstrap 抽样以扩充到 125 组数据
bootstrap_samples = original_data.sample(n=125, replace=True, random_state=42)

# 初始化 StandardScaler 用于数据标准化
scaler = StandardScaler()

# 拟合并转换扩充后的数据
standardized_data = scaler.fit_transform(bootstrap_samples)

# 将标准化后的数据转换回 DataFrame
standardized_df = pd.DataFrame(standardized_data, columns=bootstrap_samples.columns)

# 将标准化后的数据保存到 Excel 文件
standardized_df.to_excel('Standardized_Expanded_Data.xlsx', index=False)

print('扩充并标准化后的数据已保存至 "Standardized_Expanded_Data.xlsx"')
