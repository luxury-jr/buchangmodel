import matplotlib.pyplot as plt
# 示例数据
sample_sizes = [25, 50, 75, 100, 125]
# z-score规范化的 MAE
mae_nn_z = [1.1741, 0.8789, 0.9135, 0.6665, 0.7816]
mae_rf_z = [0.9326, 0.5794, 0.2905, 0.1378, 0.0855]
mae_svm_z = [1.3120, 0.9493, 1.0027, 0.6858, 0.8174]
mae_adaboost_z = [0.9132, 0.5760, 0.3493, 0.3573, 0.3506]
mae_dt_z = [0.7621, 0.5169, 0.0948, 0.0694, 0.0]
mae_gp_z = [0.6713, 0.2347, 0.0990, 0.0230, 0.0]
mae_gb_z = [0.8035, 0.4202, 0.1089, 0.0814, 0.0271]

# 绘制 z-score 规范化图表
plt.figure(figsize=(12, 6))
plt.plot(sample_sizes, mae_nn_z, label='Neural Network - z-score', marker='o')
plt.plot(sample_sizes, mae_rf_z, label='Random Forest - z-score', marker='o')
plt.plot(sample_sizes, mae_svm_z, label='SVM - z-score', marker='o')
plt.plot(sample_sizes, mae_adaboost_z, label='AdaBoost - z-score', marker='o')
plt.plot(sample_sizes, mae_dt_z, label='Decision Tree - z-score', marker='o')
plt.plot(sample_sizes, mae_gp_z, label='Gaussian Process - z-score', marker='o')
plt.plot(sample_sizes, mae_gb_z, label='Gradient Boosting - z-score', marker='o')
plt.xlabel('Sample Size')
plt.ylabel('MAE')
plt.title('Model MAE vs. Sample Size (z-score Normalization)')
plt.legend()
plt.show()
