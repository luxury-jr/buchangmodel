import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the data (Replace 'your_data_file.xlsx' with the actual file path)
data_transposed = pd.read_excel('Transposed_扩充数据_Bootstrap.xlsx')

# Preparing the data
X = data_transposed[['料管温度第一段', '保压压力第一段', '保压压力第二段', '射出速度第四段', '分段位置第四段', '保压时间一', '保压时间二', '保压时间三', '冷却时间', '上长S', '下长S', '左宽S', '右宽S']]
y = data_transposed[['上长T', '下长T', '左宽T', '右宽T']]

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Neural Network Regressor
nn_regressor = MLPRegressor(random_state=42)
nn_regressor.fit(X_train, y_train)
nn_predictions = nn_regressor.predict(X_test)

# Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_train, y_train)
rf_predictions = rf_regressor.predict(X_test)

# Support Vector Machine (SVM) Regressor with MultiOutputRegressor
svm_regressor = MultiOutputRegressor(SVR())
svm_regressor.fit(X_train, y_train)
svm_predictions = svm_regressor.predict(X_test)

# AdaBoost Regressor with MultiOutputRegressor
adaboost_regressor = MultiOutputRegressor(AdaBoostRegressor(random_state=42))
adaboost_regressor.fit(X_train, y_train)
adaboost_predictions = adaboost_regressor.predict(X_test)

# Decision Tree Regressor
decision_tree_regressor = DecisionTreeRegressor(random_state=42)
decision_tree_regressor.fit(X_train, y_train)
decision_tree_predictions = decision_tree_regressor.predict(X_test)

# Gaussian Process Regressor
gpr_regressor = GaussianProcessRegressor(kernel=RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-10), random_state=42)
gpr_regressor.fit(X_train, y_train)
gpr_predictions, gpr_std = gpr_regressor.predict(X_test, return_std=True)

# Gradient Boosting Regressor with MultiOutputRegressor
gradient_boosting_regressor = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
gradient_boosting_regressor.fit(X_train, y_train)
gradient_boosting_predictions = gradient_boosting_regressor.predict(X_test)

# Evaluation Metrics
nn_mae = mean_absolute_error(y_test, nn_predictions)
nn_mse = mean_squared_error(y_test, nn_predictions)
nn_r2 = r2_score(y_test, nn_predictions)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

svm_mae = mean_absolute_error(y_test, svm_predictions)
svm_mse = mean_squared_error(y_test, svm_predictions)
svm_r2 = r2_score(y_test, svm_predictions)

adaboost_mae = mean_absolute_error(y_test, adaboost_predictions)
adaboost_mse = mean_squared_error(y_test, adaboost_predictions)
adaboost_r2 = r2_score(y_test, adaboost_predictions)

decision_tree_mae = mean_absolute_error(y_test, decision_tree_predictions)
decision_tree_mse = mean_squared_error(y_test, decision_tree_predictions)
decision_tree_r2 = r2_score(y_test, decision_tree_predictions)

gpr_mae = mean_absolute_error(y_test, gpr_predictions)
gpr_mse = mean_squared_error(y_test, gpr_predictions)
gpr_r2 = r2_score(y_test, gpr_predictions)

gradient_boosting_mae = mean_absolute_error(y_test, gradient_boosting_predictions)
gradient_boosting_mse = mean_squared_error(y_test, gradient_boosting_predictions)
gradient_boosting_r2 = r2_score(y_test, gradient_boosting_predictions)

print("Neural Network Regressor - MAE:", nn_mae, "MSE:", nn_mse, "R2:", nn_r2)
print("Random Forest Regressor - MAE:", rf_mae, "MSE:", rf_mse, "R2:", rf_r2)
print("Support Vector Machine (SVM) Regressor - MAE:", svm_mae, "MSE:", svm_mse, "R2:", svm_r2)
print("AdaBoost Regressor - MAE:", adaboost_mae, "MSE:", adaboost_mse, "R2:", adaboost_r2)
print("Decision Tree Regressor - MAE:", decision_tree_mae, "MSE:", decision_tree_mse, "R2:", decision_tree_r2)
print("Gaussian Process Regressor - MAE:", gpr_mae, "MSE:", gpr_mse, "R2:", gpr_r2)
print("Gradient Boosting Regressor - MAE:", gradient_boosting_mae, "MSE:", gradient_boosting_mse, "R2:", gradient_boosting_r2)
