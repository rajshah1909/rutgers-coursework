# ============================================================
# HW05 — Problem 4.2: Residual Analysis and Model Adequacy
# Name: RAJ SHAH
# Date: 10/19/2025
# ============================================================

# ------------------------------------------------------------
# 1. Set Working Directory
# ------------------------------------------------------------
setwd("C:/Users/RAJ RUTGERS/Desktop/Regression Methods/HW05")

# ------------------------------------------------------------
# 2. Load Required Libraries
# ------------------------------------------------------------
library(olsrr)
library(car)

# ------------------------------------------------------------
# 3. Load Dataset
# ------------------------------------------------------------
p3_1 <- read.csv("B1.csv", stringsAsFactors = FALSE)

# ------------------------------------------------------------
# 4. Redirect Output to Text File
# ------------------------------------------------------------
sink("HW05_output.txt")

cat("============================================================\n")
cat("HW05 — Problem 4.2: Residual Analysis and Model Adequacy\n")
cat("Name: RAJ SHAH\n")
cat("============================================================\n\n")

# ============================================================
# (a) Model Summary and ANOVA
# ============================================================
cat("(a) Fit the model: y ~ x2 + x7 + x8 and show summary and ANOVA.\n\n")

lmod <- lm(y ~ x2 + x7 + x8, data = p3_1)
summary_lmod <- summary(lmod)
anova_lmod <- anova(lmod)

print(summary_lmod)
cat("\n------------------------------------------------------------\n")
cat("ANOVA Table\n")
cat("------------------------------------------------------------\n")
print(anova_lmod)

cat("\nInterpretation:\n")
cat("• The overall F-test is significant (p < 0.001).\n")
cat("• Predictors x2, x7, and x8 all have p-values < 0.05, indicating significance.\n")
cat("• R² ≈ 0.786 → model explains about 78.6% of variation in y.\n\n")

# ============================================================
# (b) Residual Normality Check (Q-Q Plot and Correlation Test)
# ============================================================
cat("(b) Residual Normality Check: Q-Q Plot and Correlation Test\n")
png("HW05_QQ_Plot.png", width = 800, height = 600)
ols_plot_resid_qq(lmod)
dev.off()

cat("\nNormality Test using ols_test_correlation:\n\n")
cor_output <- capture.output(ols_test_correlation(lmod))
cat(paste(cor_output, collapse = "\n"), "\n")
writeLines(cor_output, "HW05_Normality_Correlation.txt")

cat("\nInterpretation:\n")
cat("• If residual points follow the 45° line on QQ plot → normality reasonable.\n")
cat("• High correlation (~0.98) and p-value > 0.05 confirm normal residuals.\n\n")

# ============================================================
# (c) Residuals vs Regressors
# ============================================================
cat("============================================================\n")
cat("(c) Residuals vs Each Regressor (x2, x7, x8)\n")
cat("============================================================\n\n")

# Residuals vs fitted
png("HW05_Residuals_vs_Fitted.png", width = 800, height = 600)
plot(fitted(lmod), residuals(lmod),
     xlab = "Fitted Values", ylab = "Residuals",
     main = "Residuals vs Fitted Values")
abline(h = 0, col = "red", lty = 2)
dev.off()

# Residuals vs each regressor
png("HW05_Residuals_vs_x2.png", width = 800, height = 600)
plot(p3_1$x2, residuals(lmod),
     xlab = "x2", ylab = "Residuals", main = "Residuals vs x2")
abline(h = 0, col = "red", lty = 2)
dev.off()

png("HW05_Residuals_vs_x7.png", width = 800, height = 600)
plot(p3_1$x7, residuals(lmod),
     xlab = "x7", ylab = "Residuals", main = "Residuals vs x7")
abline(h = 0, col = "red", lty = 2)
dev.off()

png("HW05_Residuals_vs_x8.png", width = 800, height = 600)
plot(p3_1$x8, residuals(lmod),
     xlab = "x8", ylab = "Residuals", main = "Residuals vs x8")
abline(h = 0, col = "red", lty = 2)
dev.off()

cat("Interpretation:\n")
cat("• Residuals appear randomly scattered around zero.\n")
cat("• No strong curvature → linear relationship is adequate.\n")
cat("• No funnel shape → constant variance assumption holds.\n\n")

# ============================================================
# (d) Added Variable (Partial Regression) Plots
# ============================================================
cat("============================================================\n")
cat("(d) Added Variable (Partial Regression) Plots\n")
cat("============================================================\n\n")

# Combined AV plots (from 'car' package)
png("HW05_Added_Variable_Plots_CAR.png", width = 900, height = 700)
avPlots(lmod, ask = FALSE, id.n = 0,
        main = "Added Variable (Partial Regression) Plots — car package")
dev.off()

# Combined AV plots (olsrr new version: prints all at once)
png("HW05_Added_Variable_Plots_OLS.png", width = 900, height = 700)
ols_plot_added_variable(lmod, print_plot = TRUE)
dev.off()

cat("Interpretation:\n")
cat("• Each Added Variable Plot shows the effect of one predictor after removing others.\n")
cat("• x2 has a strong positive slope (largest effect).\n")
cat("• x7 is moderately positive.\n")
cat("• x8 is clearly negative, confirming its inverse relation.\n")
cat("• No strong outliers or curvature are present → model form appropriate.\n\n")

# ============================================================
# End of HW05 Output
# ============================================================
sink()
graphics.off()

cat("✅ All results and plots successfully saved to:\n")
cat("C:/Users/RAJ RUTGERS/Desktop/Regression Methods/HW05\n")
cat("✅ Script executed successfully!\n")
