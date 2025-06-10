import pandas as pd
import numpy as np
from lifelines import CoxPHFitter


# 示例数据集
data = pd.read_excel('代谢免疫综合指标 - 对齐UKB_result（测试） - 分层分析.xlsx', sheet_name='Female')

# 确保数据中有'status'和'time'列，以及你关注的主变量和协变量
status_column = 'cardiovascular_death_status'
time_column = 'survival_time'
main_variable = 'Comprehensive indicators  category'
covariates = ['age','race','marriage','education','smoking','drinking',
              'BMI','Hypertension','Hyperlipidemia','HbA1c','HDL']  # 替换为实际协变量列名

# 提取所需列
df = data[[time_column, status_column, main_variable] + covariates]

# 处理缺失值：可以选择填充或删除缺失值
df = df.dropna()  # 或者使用 df.fillna(df.mean()) 填充缺失值

# 初始化CoxPHFitter
cph = CoxPHFitter()  # 添加惩罚项帮助模型收敛

# 拟合Cox模型
cph.fit(df, duration_col=time_column, event_col=status_column)

# 打印Cox回归模型的总结信息
cph.print_summary()

# 获取主变量的HR值及其置信区间
hr = cph.hazard_ratios_[main_variable]
conf_int = cph.confidence_intervals_.loc[main_variable]
ci_lower = np.exp(conf_int.iloc[0])
ci_upper = np.exp(conf_int.iloc[1])
p_value = cph.summary.loc[main_variable, 'p']

# 检查 p 值是否小于 0.001，并输出相应信息
if p_value < 0.001:
    print("p-value<0.001, actual value:", f"{p_value:.6f}")
else:
    print(f"p-value = {p_value:.6f}")

# 输出结果
print(f"HR for {main_variable}: {hr:3f}")
print(f"95% CI for {main_variable}: {ci_lower:3f}, {ci_upper:3f}")
