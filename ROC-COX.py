import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

# 示例数据集
data = pd.read_csv('代谢免疫综合指标 - 对齐UKB_result（测试）.csv')
# 确保数据中有'status'和'time'列，以及你关注的主变量和协变量
status_column = 'cardiovascular_death_status'
time_column = 'survival_time'
main_variable = 'comprehensive_index'
covariates = ['age','gender','race','marriage','education','smoking','drinking',
              'BMI','Hypertension','Hyperlipidemia','HbA1c','HDL']  # 替换为实际协变量列名

# 提取所需列
df = data[[time_column, status_column, main_variable] + covariates]

# 初始化CoxPHFitter
cph = CoxPHFitter()  # ,, show_progress=True,penalizer=0.01

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

print(f"HR for {main_variable}: {hr:3f}")
print(f"95% CI for {main_variable}: {ci_lower:3f}, {ci_upper:3f}")

"""
遍历导入文件的所有特征，计算每个特征分别和协变量的HR值
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

# 示例数据集
data = pd.read_csv(r'要计算cox的文件名.csv', engine='python', encoding='gb18030')

# 确保数据中有'status'和'time'列，以及协变量列
status_column = 'status'
time_column = 'time'
covariates = ['age', 'gender', 'race', 'marriage', 'education', 'smoking', 'drinking']  # 替换为实际协变量列名

# 需要排除的列
excluded_columns = ['CHF', 'HA','SDDSRVYR','SEQN'] + covariates + [status_column, time_column]

# 获取所有列的名称
all_columns = data.columns

# 过滤出主变量（即所有非协变量和排除变量的列）
main_variables = [col for col in all_columns if col not in excluded_columns]
# 初始化CoxPHFitter
cph = CoxPHFitter()

# 循环遍历每一个主变量
for main_variable in main_variables:
    print(f"Processing variable: {main_variable}")

    # 提取当前主变量和协变量所需列
    df = data[[time_column, status_column, main_variable] + covariates]

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
        print("p-value<0.001.")
    else:
        print("p-value>0.001.")

    print(f"HR for {main_variable}: {hr:.6f}")
    print(f"95% CI for {main_variable}: {ci_lower:.6f}, {ci_upper:.6f}")
    print("-" * 50)  # 分隔线
"""