import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib

# ------------------ 数据准备 ------------------
df = pd.read_csv('代谢免疫综合指标 - 对齐UKB.csv', encoding='utf-8')

# 明确定义特征类型
numerical_features = [
    'Lymphocyte %', 'Monocyte %', 'Neutrophil %', 'Basophil %', 'Neutrophil count',
    'Lymphocyte count', 'Albumin', 'CRP', 'BMI', 'RBC','Hb', 'MCV', 'MCH', 'RDW', 'MPV',
    'TC', 'HDL', 'HbA1c', 'Urea nitrogen', 'Ca', 'GGT', 'uric acid', 'creatinine',
]

categorical_features = [
    'Hypertension',  # 二分类 (0/1)
    'Hyperlipidemia',  # 二分类 (0/1)
    'smoking',  # 三分类 (0/1/2)
    'drinking'  # 二分类 (0/1)
]

# 目标变量
X = df[numerical_features + categorical_features]
y = df['all_cause_death_status']

# ------------------ 数据预处理 ------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(with_mean=False), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='drop'
)

# ------------------ 核心流水线定义 ------------------
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('selector', SelectFromModel(
        LassoCV(cv=10, random_state=42, max_iter=10000, eps=1e-4, n_alphas=100),
        max_features=5
    )),
    ('smote', SMOTE(
        random_state=42,
        sampling_strategy=0.6,
        k_neighbors=3
    )),
    ('classifier', LogisticRegression(
        random_state=42,
        max_iter=10000,
        solver='liblinear',
        fit_intercept=False,
        class_weight='balanced'
    ))
])

# ------------------ 参数调优设置 ------------------
param_grid = {
    'smote__sampling_strategy': [0.5, 0.6, 0.7],
    'classifier__C': np.logspace(-4, 2, 20),
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight': [None, 'balanced']
}

# ------------------ 交叉验证设置 ------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ------------------ 模型训练 ------------------
print("开始模型训练...")
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=2
)
grid_search.fit(X, y)

# ------------------ 结果评估 ------------------
print("\n最佳参数组合：")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_

# 交叉验证预测
y_pred_proba = cross_val_predict(
    best_model, X, y,
    cv=cv,
    method='predict_proba',
    n_jobs=-1
)[:, 1]

# 计算AUC
fpr, tpr, _ = roc_curve(y, y_pred_proba)
roc_auc = auc(fpr, tpr)
print(f"\n交叉验证AUC: {roc_auc:.4f}")

# ------------------ 特征重要性解析 ------------------
preprocessor = best_model.named_steps['preprocessor']
selector = best_model.named_steps['selector']
classifier = best_model.named_steps['classifier']

selected_indices = np.where(selector.get_support())[0]
all_feature_names = preprocessor.get_feature_names_out()
selected_features = all_feature_names[selected_indices]
scaled_coefficients = classifier.coef_[0]

# 反标准化数值特征系数
scaler = preprocessor.named_transformers_['num']
numerical_scale = scaler.scale_

selected_scales = []
for feat_name in selected_features:
    if feat_name.startswith('num__'):
        original_feat = feat_name.split('__', 1)[1]
        scale_idx = numerical_features.index(original_feat)
        selected_scales.append(numerical_scale[scale_idx])
    else:
        selected_scales.append(1.0)

unscaled_coefficients = scaled_coefficients / selected_scales

# 生成预测公式
formula = " + ".join(
    [f"({coef:.5f}*{feat.split('__', 1)[1]})" for coef, feat in zip(unscaled_coefficients, selected_features)])
print(f"\n综合预测公式：\nScore = {formula}")

# 保存模型参数
feature_params = {
    'preprocessor': preprocessor,
    'selected_features': selected_features.tolist(),
    'scaled_coefficients': scaled_coefficients.tolist(),
    'unscaled_coefficients': unscaled_coefficients.tolist(),
    'intercept': 0.0
}
joblib.dump(feature_params, 'model_params.pkl')


# ------------------ 可视化 ------------------
def plot_roc_curve(y_true, y_proba, features_df, coefficients, title_suffix=""):
    plt.figure(figsize=(10, 8))

    # 模型整体ROC
    fpr_model, tpr_model, _ = roc_curve(y_true, y_proba)
    roc_auc_model = auc(fpr_model, tpr_model)
    plt.plot(fpr_model, tpr_model, color='darkorange', lw=2,
             label=f'Model ROC (AUC = {roc_auc_model:.2f})')

    # 各特征ROC
    for i, feature in enumerate(features_df.columns):
        feature_data = features_df[feature]
        coef_sign = np.sign(coefficients[i])
        adjusted_data = feature_data * coef_sign

        fpr_feat, tpr_feat, _ = roc_curve(y_true, adjusted_data)
        roc_auc_feat = auc(fpr_feat, tpr_feat)
        plt.plot(fpr_feat, tpr_feat, lw=1,
                 label=f'{feature.split("__")[1]} ROC (AUC = {roc_auc_feat:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve Comparison {title_suffix}')
    plt.legend(loc="lower right", prop={'size': 8})
    plt.savefig(f'roc_comparison_{title_suffix}.png', dpi=300, bbox_inches='tight')
    plt.close()


# 训练集可视化
X_processed = preprocessor.transform(X)
X_processed_df = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())
selected_features_df = X_processed_df[selected_features]
plot_roc_curve(y, y_pred_proba, selected_features_df, scaled_coefficients, "Train")


# ------------------ 外部验证函数 ------------------
def external_validation(data_path):
    try:
        params = joblib.load('model_params.pkl')
        preprocessor = params['preprocessor']
        selected_features = params['selected_features']
        scaled_coefficients = np.array(params['scaled_coefficients'])

        df_ext = pd.read_csv(data_path, encoding='utf-8')
        X_processed = preprocessor.transform(df_ext)
        X_processed_df = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())

        missing = set(selected_features) - set(X_processed_df.columns)
        if missing:
            raise ValueError(f"缺少必要特征：{missing}")

        # 计算得分和概率
        X_selected = X_processed_df[selected_features]
        scores = X_selected.dot(scaled_coefficients)
        probabilities = 1 / (1 + np.exp(-scores))

        df_ext['comprehensive_score'] = scores
        df_ext['probability'] = probabilities

        # 如果有真实标签则绘制ROC
        if 'all_cause_death_status' in df_ext.columns:
            y_ext = df_ext['all_cause_death_status']
            fpr_ext, tpr_ext, _ = roc_curve(y_ext, df_ext['probability'])
            auc_ext = auc(fpr_ext, tpr_ext)
            print(f"外部验证AUC：{auc_ext:.4f}")

            # 外部验证可视化
            selected_features_df = X_processed_df[selected_features]
            plot_roc_curve(y_ext, probabilities, selected_features_df, scaled_coefficients, "Validation")

            # 保存验证结果
            output_path = data_path.replace('.csv', '_validation_results.csv')
            df_ext.to_csv(output_path, index=False)
            print(f"验证结果已保存至 {output_path}")

        # 保存预测结果
        output_path = data_path.replace('.csv', '_result.csv')
        df_ext.to_csv(output_path, index=False)

        return df_ext
    except Exception as e:
        print(f"外部验证失败：{str(e)}")
        raise


# 执行验证
external_data_path = 'UKB数据-对齐NHANES(外部验证).csv'
external_result = external_validation(external_data_path)