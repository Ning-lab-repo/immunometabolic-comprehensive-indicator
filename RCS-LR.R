# 加载必要的包
library(rms)
library(ggplot2)

# 读取数据
df <- read.csv("F:\\Python-study1\\UKB数据-对齐NHANES(外部验证)_result（测试）.csv")

# 数据预处理 ----------------------------------------------------------------
# 检查关键变量是否存在缺失值
cat("缺失值检查:\n")
print(colSums(is.na(df[c("Cardiovascular_disease", "comprehensive_index", "age", "gender")])))
# 设置数据分布环境
dd <- datadist(df)
options(datadist = 'dd')

# 模型构建 ----------------------------------------------------------------
fit <- lrm(Congestive_heart_failure ~ rcs(comprehensive_index, 5) + 
             age + gender + smoking + drinking + BMI + 
             Hypertension + Hyperlipidemia + HbA1c + HDL, 
           data = df, x = TRUE, y = TRUE)

# 模型诊断
cat("\n模型拟合效果:\n")
print(fit)

# 非线性检验 --------------------------------------------------------------
anova_fit <- anova(fit)
cat("\n非线性检验结果:\n")
print(anova_fit)

# 安全提取P值 ------------------------------------------------------------
# 方法1：通过行名模式匹配
p_row <- grep("comprehensive_index", rownames(anova_fit), value = TRUE)
if(length(p_row) > 0) {
  p_value <- anova_fit[p_row[1], "P"]
} else {
  # 方法2：通过列特征匹配（最后一行通常是非线性检验）
  p_value <- anova_fit[nrow(anova_fit), ncol(anova_fit)]
}

# 处理异常情况
if (is.null(p_value)) {
  warning("未能提取到有效的P值，请手动检查anova结果")
  p_value <- 1
}

# OR值计算 ----------------------------------------------------------------
OR <- Predict(fit, 
              comprehensive_index = seq(min(df$comprehensive_index), 
                                        max(df$comprehensive_index), 
                                        length = 200),
              fun = exp, 
              ref.zero = TRUE)

# 精确查找OR=1的点（使用线性插值） ------------------------------------------
find_ref_point <- function(pred) {
  below <- which(pred$yhat < 1)
  above <- which(pred$yhat > 1)
  
  if (length(below) == 0 | length(above) == 0) return(NA)
  
  cross_index <- max(below[below < min(above)])
  
  x0 <- pred$comprehensive_index[cross_index]
  x1 <- pred$comprehensive_index[cross_index+1]
  y0 <- pred$yhat[cross_index] - 1
  y1 <- pred$yhat[cross_index+1] - 1
  
  a <- (y1 - y0)/(x1 - x0)
  b <- y0 - a*x0
  exact_x <- -b/a
  
  return(exact_x)
}
ref_point <- find_ref_point(OR)
cat("\nOR=1的参考点:", round(ref_point, 3), "\n")

# 可视化 ------------------------------------------------------------------
p_text <- ifelse(p_value < 0.001, 
                 "P for nonlinearity < 0.001",
                 paste("P for nonlinearity =", round(p_value, 3)))

p <- ggplot(OR, aes(x = comprehensive_index, y = yhat)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), 
              alpha = 0.2, fill = "#1E90FF") +
  geom_line(color = "#1E90FF", linewidth = 1.2) +
  geom_hline(yintercept = 1, linetype = 2, color = "gray30") +
  geom_vline(xintercept = ref_point, linetype = 3, color = "red") +
  annotate("text", x = ref_point, y = max(OR$yhat)*1.1,
           label = paste("Reference:", round(ref_point, 2)),
           color = "red", hjust = -0.1) +
  labs(title = "Restricted Cubic Spline Analysis",
       subtitle = p_text,
       x = "Comprehensive Index (Standardized)",
       y = "Odds Ratio (95% CI)") +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid.major = element_line(color = "grey90"),
    panel.grid.minor = element_blank(),
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, color = "gray30"),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black")
  )

# 打印图形
print(p)

# 保存图形
ggsave("RCS_Analysis_Plot.png", p, width = 10, height = 6, dpi = 300)


