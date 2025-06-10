import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc

# 读取数据
data = pd.read_excel('F:/Python-study1/UKB数据-对齐NHANES(外部验证)_人群分布测试.xlsx', sheet_name='威尔士')

# 定义分析变量
main_variable = 'MCH'
diseases = [
    'Congestive heart failure',
    'Coronary heart disease',
    'Angina pectoris',
    'Heart attack',
    'Stroke',
    'Cardiovascular disease'
]

covariates = ['age', 'gender', 'smoking', 'drinking',
              'BMI', 'Hypertension', 'Hyperlipidemia', 'HbA1c', 'HDL']

# 初始化绘图
plt.figure(figsize=(10, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(diseases)))  # 使用tab10调色板

# 结果存储字典
results = {}

# 循环分析每个疾病
for idx, disease in enumerate(diseases):
    # 数据预处理
    df = data[[disease, main_variable] + covariates].dropna()
    df = pd.get_dummies(df, drop_first=True)

    # 准备模型数据
    X = df.drop(columns=[disease])
    X = sm.add_constant(X)
    y = df[disease]

    # 逻辑回归
    try:
        logit_model = sm.Logit(y, X)
        result = logit_model.fit(disp=0)
    except:
        print(f"无法收敛: {disease}")
        continue

    # 保存结果
    results[disease] = {
        'model': result,
        'OR': np.exp(result.params[main_variable]),
        'CI_lower': np.exp(result.conf_int().loc[main_variable][0]),
        'CI_upper': np.exp(result.conf_int().loc[main_variable][1]),
        'p_value': result.pvalues[main_variable]
    }

    # 计算ROC曲线
    y_pred = result.predict(X)
    fpr, tpr, _ = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)

    # 绘制曲线
    plt.plot(fpr, tpr, color=colors[idx],
             label=f'{disease} (AUC={roc_auc:.2f})')

# 绘制参考线
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('Combined_ROC_Curves.png', dpi=300)


# 打印汇总结果
print("\n综合指数与各疾病的关联性分析：")
for disease, res in results.items():
    print(f"\n{disease}:")
    print(f"OR值: {res['OR']:.2f} (95%CI: {res['CI_lower']:.2f}-{res['CI_upper']:.2f})")
    print(f"P值: {res['p_value']:.4f}")