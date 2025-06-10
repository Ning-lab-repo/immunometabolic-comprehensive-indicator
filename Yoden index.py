import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use('TkAgg')  # 或者试试 'Qt5Agg'

# 读取文件（确保文件包含 comprehensive_index 和 actual_label 两列）
data = pd.read_csv('UKB数据-对齐NHANES(外部验证)_result（测试）.csv')


# 检查数据结构
if 'comprehensive_index' not in data.columns or 'all_cause_death_status' not in data.columns:
    raise ValueError("文件中必须包含 'comprehensive_index' 和 'all_cause_death_status' 列")

# 提取预测分数和真实标签
y_scores = data['comprehensive_index']
y_true = data['all_cause_death_status']

# 计算灵敏度和特异性
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# 计算约登指数
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]

# 输出最佳阈值和相关信息
print(f"最佳阈值: {optimal_threshold}")
print(f"灵敏度: {tpr[optimal_idx]:.4f}, 特异性: {1 - fpr[optimal_idx]:.4f}")
import matplotlib.pyplot as plt

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label="ROC Curve")
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f"Optimal Threshold: {optimal_threshold:.2f}")
plt.title("ROC Curve with Optimal Threshold")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.legend()
plt.grid()
plt.savefig('roc_curve.png')

