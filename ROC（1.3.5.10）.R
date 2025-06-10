library(survival)
library(timeROC)
library(ggplot2)
library(scales)  # 用于alpha透明度函数
library(readxl)

df <- read_excel("F:\\Python-study1\\UKB数据-对齐NHANES(外部验证)_人群分布测试.xlsx", sheet = 1)

# 构建生存对象
surv_obj <- Surv(df$survival_time, df$cardiovascular_death_status)

# 原始列名（保持与数据框一致）
features <- c("comprehensive_index", "Lymphocyte", "Albumin", "RBC", "Hb", "MCH")

# 显示标签（允许使用特殊字符）
feature_display <- c(
  "Comprehensive indicator", 
  "Lymphocyte%", 
  "Albumin", 
  "RBC", 
  "Hb", 
  "MCH"
)

# 时间点设置
time_points <- c(12, 36, 60, 120)

# 计算时间依赖性ROC
roc_results <- lapply(features, function(feature) {
  timeROC(
    T = df$survival_time,
    delta = df$cardiovascular_death_status,
    marker = df[[feature]],
    cause = 1,
    times = time_points,
    iid = TRUE
  )
})

# ROC曲线调整函数
adjust_roc_curve <- function(roc_obj, time_index = 1) {
  tp <- roc_obj$TP[, time_index]
  fp <- roc_obj$FP[, time_index]
  auc_val <- roc_obj$AUC[time_index]
  
  if(auc_val < 0.5) {
    tp <- 1 - tp
    fp <- 1 - fp
    ord <- order(fp)
  } else {
    ord <- order(fp)
  }
  
  list(
    FP = fp[ord],
    TP = tp[ord],
    AUC = ifelse(auc_val < 0.5, 1 - auc_val, auc_val)
  )
}

# 调整置信区间计算
adjust_auc_ci <- function(roc_obj, time_index = 1) {
  auc_val <- roc_obj$AUC[time_index]
  ci_matrix <- confint(roc_obj)$CI_AUC
  
  if(auc_val < 0.5) {
    new_lower <- 1 - (ci_matrix[time_index, 2]/100)
    new_upper <- 1 - (ci_matrix[time_index, 1]/100)
  } else {
    new_lower <- ci_matrix[time_index, 1]/100
    new_upper <- ci_matrix[time_index, 2]/100
  }
  
  list(
    lower = max(0.5, new_lower),
    upper = min(1.0, new_upper)
  )
}

# 主绘图参数设置
colors <- c("#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B")
plot_colors <- setNames(colors, features)  # 仍使用原始列名映射颜色

# 绘制12个月ROC曲线
par(mar = c(4.5, 4.5, 2, 1))
plot(NULL, xlim = c(0,1), ylim = c(0,1), 
     xlab = "False Positive Rate", ylab = "True Positive Rate",
     main = "Time-Dependent ROC Curves at 12 Months")
abline(0, 1, col = "gray60", lty = 2)

adjusted_curves <- lapply(seq_along(features), function(i) {
  curve_data <- adjust_roc_curve(roc_results[[i]], 1)
  lines(curve_data$FP, curve_data$TP, 
        col = plot_colors[features[i]], lwd = 2)  # 使用原始列名索引颜色
  points(curve_data$FP, curve_data$TP, 
         col = alpha(plot_colors[features[i]], 0.5), pch = 19, cex = 0.6)
  curve_data
})

# 生成图例标签（关键修改点）
legend_labels <- sapply(seq_along(features), function(i) {
  res <- list(
    auc = adjusted_curves[[i]]$AUC,
    ci = adjust_auc_ci(roc_results[[i]], 1)
  )
  sprintf("%s: %.2f (%.2f-%.2f)", 
          feature_display[i],  # 使用显示标签
          res$auc, 
          res$ci$lower, 
          res$ci$upper)
})

# 添加图例
legend("bottomright", 
       legend = legend_labels,
       col = plot_colors[features],  # 颜色仍按原始顺序
       lwd = 2, cex = 0.7,
       bg = "white", box.col = "gray80")

