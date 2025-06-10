import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  #
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc

# 读取数据
data = pd.read_excel('F:/Python-study1/UKB数据-对齐NHANES(外部验证)_人群分布测试.xlsx', sheet_name='苏格兰')



# 确保数据中有'cardiovascular disease'和'comprehensive_index'列
status_column = 'Cardiovascular disease'
main_variable = 'comprehensive_index'
covariates = ['age','gender','smoking','drinking',
              'BMI','Hypertension','Hyperlipidemia','HbA1c','HDL']  # 替换为实际的协变量列名

# 提取相关列
df = data[[status_column, main_variable] + covariates]

# 将分类变量转化为哑变量（如果存在分类变量）
df = pd.get_dummies(df, drop_first=True)

# 分离特征和目标变量
X = df.drop(columns=[status_column])
y = df[status_column]

# 添加常数项
X_const = sm.add_constant(X)

# 使用statsmodels的Logit模型进行逻辑回归（包括comprehensive_index和协变量）
logit_model = sm.Logit(y, X_const)
result = logit_model.fit()

# 打印回归结果
print(result.summary())

# 获取回归系数、赔率比（OR）及其95%置信区间
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': result.params[1:],  # 排除常数项
    'OR (Odds Ratio)': np.exp(result.params[1:]),  # 转换为赔率比
    '95% CI Lower': np.exp(result.conf_int()[0][1:]),  # 计算置信区间下限
    '95% CI Upper': np.exp(result.conf_int()[1][1:])  # 计算置信区间上限
})

# 按照OR值排序
coef_df = coef_df.sort_values(by='OR (Odds Ratio)', ascending=False)

# 打印回归系数和置信区间
print(coef_df)

# 获取comprehensive_index的OR（赔率比）和95%置信区间
comprehensive_index_coef = result.params[main_variable]  # 获取comprehensive_index的回归系数
comprehensive_index_or = np.exp(comprehensive_index_coef)  # 转换为OR（赔率比）

# 获取comprehensive_index的95%置信区间
comprehensive_index_ci_lower = np.exp(result.conf_int().loc[main_variable][0])
comprehensive_index_ci_upper = np.exp(result.conf_int().loc[main_variable][1])

# 获取comprehensive_index的p值
comprehensive_index_p_value = result.pvalues[main_variable]

# 打印comprehensive_index的OR及其95%置信区间
print(f"Comprehensive Index OR: {comprehensive_index_or:.4f}")
print(f"95% Confidence Interval for OR: ({comprehensive_index_ci_lower:.4f}, {comprehensive_index_ci_upper:.4f})")
print(f"Comprehensive Index p-value: {comprehensive_index_p_value:.6f}")  # 输出p值

# 计算预测概率
y_pred_prob = result.predict(X_const)

# 计算ROC曲线数据
fpr, tpr, thresholds = roc_curve(y, y_pred_prob)

# 计算AUC值（曲线下面积）
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # 对角线
plt.title('ROC curve of Cardiovascular disease - Comprehensive Index')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.savefig('ROC_curve.png')


