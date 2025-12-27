# ============================================================
# HW04 — Problem 3.8: CO2 Product vs Solvent & Hydrogen
# Name: RAJ SHAH
# Date: 9/30/2025
# ============================================================

# Set working directory
setwd("C:/Users/RAJ RUTGERS/Desktop/Regression Methods/HW04")

# Load dataset
b5 <- read.csv("B5.csv", stringsAsFactors = FALSE)

# Start writing all results into HW04_output.txt
sink("HW04_output.txt")

cat("============================================================\n")
cat("HW04 — Problem 3.8: CO2 Product vs Solvent & Hydrogen\n")
cat("Name: RAJ SHAH\n")
cat("============================================================\n\n")

# ============================================================
# (a) Fit a multiple regression model: y ~ x6 + x7
# ============================================================
cat("(a) Fit a multiple regression model relating CO2 product (y) to total solvent (x6) and hydrogen consumption (x7).\n\n")
model_full <- lm(y ~ x6 + x7, data = b5)
print(summary(model_full))

# ============================================================
# (b) Test for significance of regression + R² and Adj R²
# ============================================================
cat("\n(b) Test for significance of regression. Calculate R² and Adjusted R².\n\n")
anova_result <- anova(model_full)
print(anova_result)

r2 <- summary(model_full)$r.squared
r2_adj <- summary(model_full)$adj.r.squared
cat(sprintf("\nR²: %.4f", r2))
cat(sprintf("\nAdjusted R²: %.4f\n", r2_adj))

f_stat <- summary(model_full)$fstatistic
cat(sprintf("\nF-statistic: %.3f on %d and %d DF, p-value: %.6f\n",
            f_stat[1], f_stat[2], f_stat[3],
            pf(f_stat[1], f_stat[2], f_stat[3], lower.tail = FALSE)))

cat("Conclusion: The regression is highly significant (p < 0.001). At least one regressor contributes to explaining CO2.\n")

# ============================================================
# (c) t-tests for x6 and x7
# ============================================================
cat("\n(c) Using t-tests to determine the contribution of x6 and x7.\n\n")
print(coef(summary(model_full)))

cat("\nInterpretation: Both x6 (solvent total) and x7 (hydrogen consumption) have p-values < 0.05.\n")
cat("This indicates they significantly contribute to predicting CO2.\n")

# ============================================================
# (d) 95% Confidence Intervals for β6 and β7
# ============================================================
cat("\n(d) Construct 95% CIs on β6 and β7.\n\n")
ci_full <- confint(model_full, level = 0.95)
print(ci_full)

# ============================================================
# (e) Refit using only x6, test regression, compute R² and Adj R²
# ============================================================
cat("\n(e) Refit the model using only x6 as the regressor.\n\n")
model_x6 <- lm(y ~ x6, data = b5)
print(summary(model_x6))
anova_x6 <- anova(model_x6)
print(anova_x6)

cat(sprintf("\nNew R²: %.4f", summary(model_x6)$r.squared))
cat(sprintf("\nNew Adjusted R²: %.4f\n", summary(model_x6)$adj.r.squared))

cat("\nDiscussion: The x6-only model explains less variability than the full model.\n")
cat("While still significant, it omits some explanatory power provided by x7.\n")

# ============================================================
# (f) 95% CI for β6 in reduced model + compare to full model
# ============================================================
cat("\n(f) Construct a 95% CI on β6 using reduced model. Compare with part (d).\n\n")
ci_x6 <- confint(model_x6, level = 0.95)
print(ci_x6)

len_full <- ci_full["x6",2] - ci_full["x6",1]
len_reduced <- ci_x6["x6",2] - ci_x6["x6",1]
cat(sprintf("\nLength of CI in full model: %.4f", len_full))
cat(sprintf("\nLength of CI in reduced model: %.4f\n", len_reduced))

if (len_reduced > len_full) {
  cat("Interpretation: The CI widened when x7 was removed. This shows x7 helps improve estimate precision.\n")
} else {
  cat("Interpretation: The CI did not widen much. x7 may not add strong precision.\n")
}

# ============================================================
# (g) Compare MSEs from both models
# ============================================================
cat("\n(g) Compare MS_Res (MSE) for full vs reduced models.\n\n")
#mse_full <- mean(model_full$residuals^2)
#mse_x6 <- mean(model_x6$residuals^2)

mse_full <- deviance(model_full) / df.residual(model_full)
mse_x6   <- deviance(model_x6) / df.residual(model_x6)


cat(sprintf("MSE (full model): %.4f\n", mse_full))
cat(sprintf("MSE (x6-only model): %.4f\n", mse_x6))

if (mse_x6 > mse_full) {
  cat("\nConclusion: Removing x7 increased residual error. This confirms x7 improves model fit.\n")
} else {
  cat("\nConclusion: Removing x7 did not change residual error much. x7’s contribution may be limited.\n")
}

# Stop saving to file
sink()

# ============================================================
# (Optional) Save residual histograms to PDF
# ============================================================
pdf("HW04_graphs.pdf")
hist(model_full$residuals, main = "Residuals: Full Model", col = "skyblue", xlab = "Residuals")
hist(model_x6$residuals, main = "Residuals: Reduced Model", col = "lightgreen", xlab = "Residuals")
dev.off()
