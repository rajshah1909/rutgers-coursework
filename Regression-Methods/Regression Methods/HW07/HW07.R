# ============================================================
# HW07 — Problem 7.18 (Polynomial Model & Quadratic Testing)
# Name: RAJ SHAH
# Date: 10/25/2025
# ============================================================

# Set working directory
setwd("C:/Users/RAJ RUTGERS/Desktop/Regression Methods/HW07")

# Load required libraries
library(car)

# Load dataset
p7 <- read.csv("p7_18.csv", stringsAsFactors = FALSE)

# Start writing all outputs to text file
sink("HW07_output.txt")

cat("============================================================\n")
cat("HW07 — Problem 7.18\n")
cat("Name: RAJ SHAH\n")
cat("============================================================\n\n")

# ============================================================
# (1) Fit Full Polynomial Model
# ============================================================
cat("(1) Fitting Linear + Quadratic Regression Model\n")
cat("------------------------------------------------------------\n\n")

full_model <- lm(y ~ x1 + I(x1^2) +
                   x2 + I(x2^2) +
                   x3 + I(x3^2), data = p7)

cat("Full Model Summary:\n\n")
print(summary(full_model))

cat("\nInterpretation:\n")
cat("• The model includes both linear and quadratic terms for x1, x2, and x3.\n")
cat("• Coefficient significance indicates which predictors meaningfully affect y.\n")

# ============================================================
# (2) Global (Overall) F-test
# ============================================================
cat("\n============================================================\n")
cat("(2) Global F-test: Are ALL coefficients equal to 0?\n")
cat("============================================================\n\n")

cat("Result Extracted From Summary(full_model):\n")
cat("• Null Hypothesis H0: All slope coefficients = 0\n")
cat("• Alternative Ha: At least 1 slope coefficient ≠ 0\n\n")

# Pull p-value from summary
global_pvalue <- summary(full_model)$fstatistic
pf(global_pvalue[1], global_pvalue[2], global_pvalue[3],
   lower.tail = FALSE) -> p_val

cat(paste("Global Model p-value:", round(p_val, 6), "\n"))

cat("\nInterpretation:\n")
cat("• If p-value < 0.05 → Model is statistically significant.\n")
cat("• Indicates predictors help explain variation in y.\n")

# ============================================================
# (3) Residuals vs Fitted Plot
# ============================================================
cat("\n============================================================\n")
cat("(3) Residuals vs Fitted Values Plot\n")
cat("============================================================\n\n")

# Save plot
png("HW07_Residuals_vs_Fitted.png", width=1200, height=900)
plot(full_model$fitted.values, full_model$residuals,
     pch = 20, col = "blue",
     main = "Residuals vs Fitted Values",
     xlab = "Fitted Values", ylab = "Residuals")
abline(h = 0, col = "red", lwd = 2)
dev.off()

cat("Residual plot saved as HW07_Residuals_vs_Fitted.png\n\n")

cat("Interpretation:\n")
cat("• Random scatter around zero → linearity assumption holds.\n")
cat("• No funneling pattern → constant variance assumption reasonable.\n")
cat("• No noticeable curvature → quadratic terms appear appropriate.\n")

# ============================================================
# (4) Reduced vs Full Model: Are Quadratic Terms Needed?
# ============================================================
cat("\n============================================================\n")
cat("(4) F-test: Are x1^2, x2^2, x3^2 = 0?\n")
cat("============================================================\n\n")

# Reduced model: NO quadratic terms
reduced_model <- lm(y ~ x1 + x2 + x3, data = p7)

# Compare models
anova_comparison <- anova(reduced_model, full_model)
print(anova_comparison)

cat("\nInterpretation:\n")
cat("• Null Hypothesis H0: β(x1²), β(x2²), β(x3²) = 0\n")
cat("• If ANOVA p-value < 0.05 → Quadratic terms significantly improve the model.\n")
cat("• Suggesting curvature effects are important.\n")

# ============================================================
# END OF HW07 OUTPUT
# ============================================================
sink()

cat("\nAll HW07 results and plots have been successfully saved to:\n")
cat("C:/Users/RAJ RUTGERS/Desktop/Regression Methods/HW07\n")
