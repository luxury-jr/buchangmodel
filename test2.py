import matplotlib.pyplot as plt
# 示例数据
sample_sizes = [25, 50, 75, 100, 125]

# z-score规范化的 MSE
mse_nn_z = [1.8063, 1.1595, 1.2145, 0.7319, 0.8668]
mse_rf_z = [1.3953, 0.8042, 0.2297, 0.1201, 0.0187]
mse_svm_z = [2.2381, 1.3629, 1.4199, 0.8930, 1.0504]
mse_adaboost_z = [1.7805, 0.9512, 0.2510, 0.2098, 0.1737]
mse_dt_z = [1.6043, 0.9460, 0.1910, 0.1462, 0.0000]
mse_gp_z = [0.9096, 0.2179, 0.0773, 0.0164, 0.0000]
mse_gb_z = [1.4678, 0.7424, 0.1419, 0.1044, 0.0012]

# 绘制 z-score 规范化图表
plt.figure(figsize=(12, 6))
plt.plot(sample_sizes, mse_nn_z, label='Neural Network - z-score', marker='o')
plt.plot(sample_sizes, mse_rf_z, label='Random Forest - z-score', marker='o')
plt.plot(sample_sizes, mse_svm_z, label='SVM - z-score', marker='o')
plt.plot(sample_sizes, mse_adaboost_z, label='AdaBoost - z-score', marker='o')
plt.plot(sample_sizes, mse_dt_z, label='Decision Tree - z-score', marker='o')
plt.plot(sample_sizes, mse_gp_z, label='Gaussian Process - z-score', marker='o')
plt.plot(sample_sizes, mse_gb_z, label='Gradient Boosting - z-score', marker='o')
plt.xlabel('Sample Size')
plt.ylabel('MSE')
plt.title('Model MSE vs. Sample Size (z-score Normalization)')
plt.legend()
plt.show()
