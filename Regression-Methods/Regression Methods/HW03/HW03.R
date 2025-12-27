# ============================================================
# HW03 — Problem 3.7 (Table B.4): House Price Regression
# Name: RAJ SHAH
# Date: 9/30/2025
# ============================================================

# Set working directory
setwd("C:/Users/RAJ RUTGERS/Desktop/Regression Methods/HW03")

# Load required libraries
if (!require(car)) install.packages("car", dependencies=TRUE)
library(car)

# Load dataset
b4 <- read.csv("B4.csv", stringsAsFactors = FALSE)

# Start saving output to a .txt file
sink("HW03_output.txt")

cat("============================================================\n")
cat("HW03 — Problem 3.7 (Table B.4): House Price Regression\n")
cat("Name: RAJ SHAH\n")
cat("============================================================\n\n")

# ---------------------------------------------------------------------
# (a) Fit a multiple regression model relating selling price to all nine regressors
# ---------------------------------------------------------------------
cat("(a) Fit a multiple regression model relating selling price to all nine regressors.\n\n")
model_full <- lm(y ~ x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9, data = b4)
print(summary(model_full))

# ---------------------------------------------------------------------
# (b) Test for significance of regression. What conclusions can you draw?
# ---------------------------------------------------------------------
cat("\n(b) Test for significance of regression. What conclusions can you draw?\n\n")
cat("ANOVA Table:\n")
anova_result <- anova(model_full)
print(anova_result)

f_stat <- summary(model_full)$fstatistic
cat(sprintf("\nF-statistic: %.3f on %d and %d DF, p-value: %.6f\n",
            f_stat[1], f_stat[2], f_stat[3],
            pf(f_stat[1], f_stat[2], f_stat[3], lower.tail = FALSE)))

cat("\nConclusion: The overall regression model is statistically significant (p < 0.001),\n")
cat("indicating that at least one of the predictors contributes to explaining house price.\n")

# ---------------------------------------------------------------------
# (c) Use t-tests to assess the contribution of each regressor to the model.
#     Discuss your findings.
# ---------------------------------------------------------------------
cat("\n(c) Use t-tests to assess the contribution of each regressor to the model.\n")
cat("Discuss your findings.\n\n")

coef_table <- summary(model_full)$coefficients
print(coef_table)

cat("\nInterpretation:\n")
cat("- Variables with p-values < 0.05 are significant contributors to the model.\n")
cat("- Examine especially x4 (Living space), x6 (Rooms), and others based on p-values.\n")

# ---------------------------------------------------------------------
# (d) What is the contribution of lot size and living space to the model
#     given that all of the other regressors are included?
# ---------------------------------------------------------------------
cat("\n(d) What is the contribution of lot size and living space to the model\n")
cat("given that all of the other regressors are included?\n\n")

# Full model already includes x3 (Lot size) and x4 (Living space)
# Now fit reduced model without x3 and x4
model_reduced_d <- lm(y ~ x1 + x2 + x5 + x6 + x7 + x8 + x9, data = b4)

cat("Partial F-test: Comparing Full Model (with x3, x4) vs Reduced Model (without x3, x4)\n")
anova_test <- anova(model_reduced_d, model_full)
print(anova_test)

p_val_d <- anova_test$`Pr(>F)`[2]
if (!is.na(p_val_d) && p_val_d < 0.05) {
  cat("\nConclusion: Lot size and living space contribute significantly to the model (p =", round(p_val_d, 5), ").\n")
} else {
  cat("\nConclusion: Lot size and living space do not contribute significantly to the model.\n")
}

# ---------------------------------------------------------------------
# (e) Is multicollinearity a potential problem in this model?
# ---------------------------------------------------------------------
cat("\n(e) Is multicollinearity a potential problem in this model?\n\n")
vif_values <- vif(model_full)
print(vif_values)

cat("\nInterpretation:\n")
cat("- VIF values > 5 or 10 indicate potential multicollinearity.\n")
cat("- Assess which variables may be highly correlated.\n")

# Done writing to output
sink()

# ---------------------------------------------------------------------
# (Optional) Residual histogram into PDF
# ---------------------------------------------------------------------
pdf("HW03_graphs.pdf")
hist(model_full$residuals, main = "Residuals: Full Model", xlab = "Residuals", col = "skyblue", border = "black")
dev.off()
