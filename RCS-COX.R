pkgs <- c("rms", "survival", "ggplot2", "dplyr", "readxl")
pacman::p_load(pkgs, character.only = TRUE)

# 加载数据集
df <- read.csv("F:\\Python-study1\\UKB数据-对齐NHANES(外部验证)_result（测试）.csv")

# 定义Cox模型的固定协变量
fixed_covariates <- c(
  "age", "gender", "race", "marriage", 
  "education", "smoking", "drinking", "BMI", 
  "Hypertension", "Hyperlipidemia", "HbA1c", "HDL"
)

# 数据预处理
processed_data <- df

# Cox回归分析
dd <- datadist(processed_data)
options(datadist = "dd")

# 修改点1：增加样条节点数以提高拟合精度
formula_str <- paste0("Surv(survival_time, cardiovascular_death_status) ~ rcs(comprehensive_index, nk = 5) + ", 
                      paste(fixed_covariates, collapse = " + "))
fit <- cph(as.formula(formula_str), data = processed_data, x = TRUE, y = TRUE)

# 非线性检验
anova_fit <- anova(fit)
print(anova_fit)

# 预测危险比（修改点2：增加预测点的密度）
orr <- Predict(fit, comprehensive_index = seq(min(processed_data$comprehensive_index), 
                                              max(processed_data$comprehensive_index), 
                                              length = 500), 
               fun = exp, ref.zero = TRUE)

# 修改点3：使用插值法精确查找OR=1的点
find_OR1_point <- function(pred_obj) {
  # 寻找交叉点
  below <- which(pred_obj$yhat < 1)
  above <- which(pred_obj$yhat > 1)
  
  if (length(below) == 0 | length(above) == 0) return(NA)
  
  # 找到最近的交叉对
  cross_index <- tail(below, 1)
  x0 <- pred_obj$comprehensive_index[cross_index]
  x1 <- pred_obj$comprehensive_index[cross_index+1]
  y0 <- pred_obj$yhat[cross_index]
  y1 <- pred_obj$yhat[cross_index+1]
  
  # 线性插值
  slope <- (y1 - y0)/(x1 - x0)
  intercept <- y0 - slope*x0
  exact_x <- (1 - intercept)/slope
  
  return(exact_x)
}

comprehensive_index_at_OR_1 <- find_OR1_point(orr)
print(paste("精确的HR=1点：", round(comprehensive_index_at_OR_1, 4)))

# 绘图（修改点4：添加精确参考线）
p <- ggplot(orr) +
  geom_line(aes(x = comprehensive_index, y = yhat), 
            linetype = "solid", linewidth = 1, colour = "red") +
  geom_ribbon(aes(x = comprehensive_index, ymin = lower, ymax = upper), 
              alpha = 0.9, fill = "pink") +
  geom_hline(yintercept = 1, linetype = 2, size = 0.5) +
  geom_vline(xintercept = comprehensive_index_at_OR_1, 
             linetype = 2, color = "black", size = 0.3) +
  
  ggtitle("Cox回归分析：综合指数的非线性效应") +
  theme_minimal() +
  theme(
    panel.grid.major = element_line(color = "grey90"),
    panel.grid.minor = element_blank(),
    plot.title = element_text(hjust = 0.5, face = "bold"),
    # 去除右边和上边的边框，保留左边和下边的边框
    panel.border = element_blank(),
    axis.line.x = element_line(color = "black", size = 0.1),  # 添加x轴（下边框）
    axis.line.y = element_line(color = "black", size = 0.1)  # 添加y轴（左边框）
  )

# 保存图形
plot_dir <- "C:/Users/Administrator/Desktop"
plot_filename <- paste0(plot_dir, "/cox_model_plot_improved.png")
ggsave(plot_filename, p, width = 10, height = 6, dpi = 300)

# 输出模型结果（修改点5：保存更详细的信息）
if (exists("fit")) {
  summary_dir <- "C:/Users/Administrator/Desktop/comprehensive_index"
  dir.create(summary_dir, showWarnings = FALSE)
  
  # 保存完整模型参数
  sink(paste0(summary_dir, "/full_model_summary.txt"))
  print(fit)
  sink()
  
  # 保存关键参数
  write.csv(anova_fit, paste0(summary_dir, "/anova_results.csv"))
  write.csv(orr, paste0(summary_dir, "/hazard_ratio_predictions.csv"))
}

