import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, statistics
import matplotlib
matplotlib.use('TkAgg')  # 强制使用 TkAgg 后端，这个后端通常支持图形显示
import matplotlib.pyplot as plt

# 读取文件（确保文件包含 comprehensive_index, survival_time 和 cardiovascular_death_status 列）
data = pd.read_csv('UKB数据-对齐NHANES(外部验证)_result（测试）.csv')

# 检查数据结构
required_columns = ['comprehensive_index', 'survival_time', 'all_cause_death_status']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"文件中必须包含 '{col}' 列")

# 提取预测分数和真实标签
y_scores = data['comprehensive_index']
y_true = data['all_cause_death_status']

# 设置你自定义的阈值，例如：
custom_threshold = -0.089427384# 自定义阈值，可以根据实际需要调整

# 根据自定义阈值将数据分为高组和低组
data['group'] = np.where(data['comprehensive_index'] >= custom_threshold,
                         'Higher Comprehensive indicator',
                         'Lower Comprehensive indicator')

# 输出自定义阈值
print(f"自定义阈值: {custom_threshold}")

# 初始化Kaplan-Meier生存分析模型
kmf = KaplanMeierFitter()

# 自定义颜色
color_dict = {'Higher Comprehensive indicator': 'red', 'Lower Comprehensive indicator': 'blue'}

# 设置图形
plt.figure(figsize=(10, 6))

# 对每个分组进行生存分析并绘图，使用自定义颜色
for group in data['group'].unique():
    group_data = data[data['group'] == group]

    # 拟合Kaplan-Meier生存曲线
    kmf.fit(group_data['survival_time'], event_observed=group_data['all_cause_death_status'], label=group)

    # 绘制生存曲线并设置颜色
    kmf.plot_survival_function(c=color_dict[group])

# 执行Log-rank检验
group_high = data[data['group'] == 'Higher Comprehensive indicator']
group_low = data[data['group'] == 'Lower Comprehensive indicator']

# Log-rank检验：检查两组生存曲线是否有显著差异
results = statistics.logrank_test(group_high['survival_time'], group_low['survival_time'],
                                  event_observed_A=group_high['all_cause_death_status'],
                                  event_observed_B=group_low['all_cause_death_status'])

# 输出P值，并根据其大小决定格式
p_value = results.p_value
if p_value < 0.0001:
    p_value_output = "P < 0.0001"
elif p_value > 0.05:
    p_value_output = f"P = {p_value:.4f}"
else:
    p_value_output = f"P = {p_value:.4f}"

# 设置图形标签
plt.title('Kaplan-Meier Survival Analysis of All-cause Mortality')
plt.xlabel('Survival Time')
plt.ylabel('Survival Probability')

# 增加图例并将其放置在左下角
plt.legend(title='Group', loc='lower left')

# 增加网格
plt.grid(True)

# 在图形上显示P值
plt.figtext(0.75, 0.8, f"Log-rank Test: {p_value_output}", fontsize=12, ha='center')

# 显示图形
plt.savefig('KM_curve.png')

# 额外输出P值
print(f"Log-rank检验的P值: {p_value_output}")


