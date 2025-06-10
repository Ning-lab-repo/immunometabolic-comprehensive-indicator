import pandas as pd
from scipy import stats

# 假设 data 是您的数据框
data = pd.read_csv('代谢免疫综合指标 - 对齐UKB_result（测试）.csv')

# 定义自定义分组标准
custom_cutoff = 0.043318984

# 根据自定义截止值分组
data['comprehensive_group'] = (data['comprehensive_index'] > custom_cutoff).astype(int)

# 自定义分类变量和连续变量
continuous_variables = ['age', 'BMI', 'TC', 'HDL', 'HbA1c', 'smoking',
                        'drinking', 'Urea nitrogen', 'Ca', 'GGT', 'uric acid', 'creatinine', 'CRP',
                        'Neutrophil count', 'Lymphocyte count', 'Eosinophil count', 'Basophil count',
                        'Monocyte count', 'Platelet count', 'Albumin', 'WBC']
categorical_variables = ['gender', 'race', 'marriage', 'education', 'Hypertension', 'Hyperlipidemia',
                         'Cardiovascular_disease', 'smoking', 'drinking']

# 结果字典，用于保存结果
results = {}

# 对连续变量进行正态性检验并选择T检验或U检验
for column in continuous_variables:
    if column in data.columns:
        group0 = data[data['comprehensive_group'] == 0][column].dropna()
        group1 = data[data['comprehensive_group'] == 1][column].dropna()

        if len(group0) > 0 and len(group1) > 0:
            # 正态性检验
            _, p_normal_group0 = stats.shapiro(group0)
            _, p_normal_group1 = stats.shapiro(group1)

            # 判断两组数据是否满足正态分布
            if p_normal_group0 > 0.05 and p_normal_group1 > 0.05:  # 如果p值大于0.05，说明数据符合正态分布
                # 进行T检验
                t_stat, t_p_value = stats.ttest_ind(group0, group1, equal_var=False)  # Welch T检验，假设方差不相等
                results[column] = {
                    'group_0_mean': float(group0.mean()),
                    'group_0_std': float(group0.std()),
                    'group_1_mean': float(group1.mean()),
                    'group_1_std': float(group1.std()),
                    'T-test': {'t_stat': t_stat, 'p_value': t_p_value}
                }
            else:
                # 进行U检验
                u_stat, u_p_value = stats.mannwhitneyu(group0, group1, alternative='two-sided')
                results[column] = {
                    'group_0_mean': float(group0.mean()),
                    'group_0_std': float(group0.std()),
                    'group_1_mean': float(group1.mean()),
                    'group_1_std': float(group1.std()),
                    'Mann-Whitney U': {'u_stat': u_stat, 'p_value': u_p_value}
                }
        else:
            results[column] = "Data insufficient for test"
    else:
        results[column] = "Variable not found in dataset"

# 对分类变量进行卡方检验
for column in categorical_variables:
    if column in data.columns:
        try:
            contingency_table = pd.crosstab(data['comprehensive_group'], data[column])
            chi2_stat, chi2_p, dof, expected = stats.chi2_contingency(contingency_table)

            group_counts = data.groupby('comprehensive_group')[column].value_counts().unstack().fillna(0)
            group_percentages = group_counts.div(group_counts.sum(axis=1), axis=0) * 100

            results[column] = {
                'count': group_counts,
                'percentage': group_percentages,
                'Chi-squared': {'chi2_stat': chi2_stat, 'p_value': chi2_p}
            }
        except ValueError as e:
            results[column] = f"Chi-squared test not applicable: {str(e)}"
    else:
        results[column] = "Variable not found in dataset"

# 打印结果
for key, value in results.items():
    print(f"Variable: {key}")
    print(value)
    print("\n")
