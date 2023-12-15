import matplotlib.pyplot as plt

# 示例数据
sample_sizes = [25, 50, 75, 100, 125]

# 0-1规范化的 R² 分数
r2_scores_nn_01 = [-0.1067, -0.0040, -0.1797, -0.0263, -0.1216]
r2_scores_rf_01 = [0.0956, 0.2182, 0.7325, 0.8438, 0.9818]
r2_scores_svm_01 = [-0.5347, -0.2315, -0.5368, -0.3467, -0.0823]
r2_scores_adaboost_01 = [-0.1629, 0.1345, 0.6895, 0.7341, 0.8330]
r2_scores_dt_01 = [-0.0480, 0.2948, 0.7739, 0.8093, 1.0000]
r2_scores_gp_01 = [-0.2296, 0.3011, 0.9328, 0.9761, 1.0000]
r2_scores_gb_01 = [0.0691, 0.2785, 0.8323, 0.8641, 0.9989]

# z-score规范化的 R² 分数
r2_scores_nn_z = [-0.1759, -0.1252, -0.4383, 0.0351, 0.1805]
r2_scores_rf_z = [0.0895, 0.2184, 0.7280, 0.8428, 0.9824]
r2_scores_svm_z = [-0.4576, -0.3253, -0.6834, -0.1685, 0.0090]
r2_scores_adaboost_z = [-0.1635, 0.0762, 0.7029, 0.7225, 0.8362]
r2_scores_dt_z = [-0.0480, 0.0799, 0.7739, 0.8093, 1.0000]
r2_scores_gp_z = [0.4065, 0.7890, 0.9086, 0.9787, 1.0000]
r2_scores_gb_z = [0.0419, 0.2786, 0.8323, 0.8641, 0.9989]

# 绘制 0-1 规范化图表
plt.figure(figsize=(12, 6))
plt.plot(sample_sizes, r2_scores_nn_01, label='Neural Network - 0-1', marker='o')
plt.plot(sample_sizes, r2_scores_rf_01, label='Random Forest - 0-1', marker='o')
plt.plot(sample_sizes, r2_scores_svm_01, label='SVM - 0-1', marker='o')
plt.plot(sample_sizes, r2_scores_adaboost_01, label='AdaBoost - 0-1', marker='o')
plt.plot(sample_sizes, r2_scores_dt_01, label='Decision Tree - 0-1', marker='o')
plt.plot(sample_sizes, r2_scores_gp_01, label='Gaussian Process - 0-1', marker='o')
plt.plot(sample_sizes, r2_scores_gb_01, label='Gradient Boosting - 0-1', marker='o')
plt.xlabel('Sample Size')
plt.ylabel('R2 Score')
plt.title('Model Performance vs. Sample Size (0-1 Normalization)')
plt.legend()
plt.show()

# 绘制 z-score 规范化图表
plt.figure(figsize=(12, 6))
plt.plot(sample_sizes, r2_scores_nn_z, label='Neural Network - z-score', marker='o')
plt.plot(sample_sizes, r2_scores_rf_z, label='Random Forest - z-score', marker='o')
plt.plot(sample_sizes, r2_scores_svm_z, label='SVM - z-score', marker='o')
plt.plot(sample_sizes, r2_scores_adaboost_z, label='AdaBoost - z-score', marker='o')
plt.plot(sample_sizes, r2_scores_dt_z, label='Decision Tree - z-score', marker='o')
plt.plot(sample_sizes, r2_scores_gp_z, label='Gaussian Process - z-score', marker='o')
plt.plot(sample_sizes, r2_scores_gb_z, label='Gradient Boosting - z-score', marker='o')
plt.xlabel('Sample Size')
plt.ylabel('R2 Score')
plt.title('Model Performance vs. Sample Size (z-score Normalization)')
plt.legend()
plt.show()

