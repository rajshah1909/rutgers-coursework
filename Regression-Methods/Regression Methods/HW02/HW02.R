# =========================================================
# HW02 — Simple Linear Regression (SLR): Required Outputs
# Parts:
# (a) ANOVA table & test for significance of regression
# (b) 95% CI on the slope
# (c) Percent variability explained (R-squared in %)
# (d) Histogram of residuals -> saved to PDF
# =========================================================

options(stringsAsFactors = FALSE, scipen = 8, digits = 5, width = 120)

# ---- File path ----
FILE_PATH <- "C:/Users/RAJ RUTGERS/Desktop/Regression Methods/HW02/p2_18.csv"
STUDENT_NAME <- "Raj Shah"

# ---- Load & prep ----
df <- read.csv(FILE_PATH)
stopifnot(all(c("amount", "ret_impress") %in% names(df)))
df <- df[order(df$amount), ]

# ---- Fit SLR ----
model <- lm(ret_impress ~ amount, data = df)
sm    <- summary(model)

# =========================================================
# REQUIRED FILE #1: HW02_output.txt (execution log)
# =========================================================
sink("HW02_output.txt")
cat("HW02 – Execution Log\n")
cat("Timestamp:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("Student:", STUDENT_NAME, "\n")
cat("Data file:", FILE_PATH, "\n\n")

cat("Model:\n")
print(model$call)

cat("\nModel Summary:\n")
print(sm)

# (a) ANOVA & significance
cat("\n=== Part (a): ANOVA table & test for significance of regression ===\n")
aov_tab <- anova(model)
print(aov_tab)
F_val <- aov_tab$`F value`[1]; df1 <- aov_tab$Df[1]; df2 <- aov_tab$Df[2]
p_val <- pf(F_val, df1, df2, lower.tail = FALSE)
cat(sprintf("\nF statistic = %.4f (df = %d, %d), p-value = %.6g\n", F_val, df1, df2, p_val))

# (b) 95% CI on slope
cat("\n=== Part (b): 95% Confidence Interval on the slope (amount) ===\n")
ci <- confint(model, level = 0.95)
slope_ci <- ci["amount", ]; slope_est <- coef(model)["amount"]
cat(sprintf("Slope estimate (beta_1) = %.6f\n", slope_est))
cat(sprintf("95%% CI for slope: [%.6f, %.6f]\n", slope_ci[1], slope_ci[2]))

# (c) Percent variability explained
cat("\n=== Part (c): Percent of total variability explained ===\n")
r2 <- sm$r.squared
cat(sprintf("R-squared = %.4f  ->  %.2f%%%% of variability explained by the SLR model\n",
            r2, 100*r2))

# (d) Residuals histogram
cat("\n=== Part (d): Residuals histogram ===\n")
cat("Residuals summary (numeric):\n")
print(summary(residuals(model)))
cat("Histogram is saved in HW02_plots.pdf\n")

# =========================================================
# REQUIRED FILE #2: HW02_plots.pdf (graphics only)
# =========================================================
pdf("HW02_plots.pdf")
hist(residuals(model),
     main = "HW02 (d): Residuals Histogram",
     xlab = "Residuals",
     breaks = "FD")
dev.off()
