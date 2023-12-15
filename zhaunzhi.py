import pandas as pd

# 加载Excel文件
file_path = '宇海数据.xlsx'  # 替换为您的文件路径
data = pd.read_excel(file_path)

# 转置数据
data_transposed = data.transpose()

# 查看转置后的数据
print(data_transposed.head())

# 保存转置后的数据为新的Excel文件
transposed_file_path = 'Transposed_宇海数据.xlsx'  # 指定新文件的保存路径
# data_transposed.to_excel(transposed_file_path)  # 不指定index参数或者设置为index=True
data_transposed.to_excel(transposed_file_path, index=False, header=False)  # 不指定index参数或者设置为index=True

