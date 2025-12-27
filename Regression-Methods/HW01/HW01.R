# =========================================================
# HW01 — SLR with Bands & MCI Intervals (Required Outputs)
# (a) Fit model
# (b) Significance of regression (ANOVA)
# (c) 95% confidence & prediction bands (plot)
# (d) 95% CI & PI for MCI (amount = 26.9)
# =========================================================

options(stringsAsFactors = FALSE, scipen = 8, digits = 5, width = 120)

# ---- EDIT THESE ----
FILE_PATH    <- "C:/Users/RAJ RUTGERS/Desktop/Regression Methods/HW01/p2_18.csv"
STUDENT_NAME <- "Raj Shah"

# ---- Load & prep ----
df <- read.csv(FILE_PATH)
stopifnot(all(c("amount", "ret_impress") %in% names(df)))
df <- df[order(df$amount), ]

# ---- Fit model ----
model <- lm(ret_impress ~ amount, data = df)
sm    <- summary(model)

# =========================================================
# REQUIRED FILE #1: HW01_output.txt (execution log)
# =========================================================
sink("HW01_output.txt")
cat("HW01 – Execution Log\n")
cat("Timestamp:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("Student:", STUDENT_NAME, "\n")
cat("Data file:", FILE_PATH, "\n\n")

# (a) Fit model
cat("=== Part (a): Fitted Regression Model ===\n")
coefs <- coef(model)
cat(sprintf("y_hat = %.4f + %.4f * amount\n", coefs[1], coefs[2]))
print(sm)

# (b) Significance of regression
cat("\n=== Part (b): ANOVA table & test for significance of regression ===\n")
aov_tab <- anova(model)
print(aov_tab)
F_val <- aov_tab$`F value`[1]; df1 <- aov_tab$Df[1]; df2 <- aov_tab$Df[2]
p_val <- pf(F_val, df1, df2, lower.tail = FALSE)
cat(sprintf("\nF statistic = %.4f (df = %d, %d), p-value = %.6g\n", F_val, df1, df2, p_val))
if (p_val < 0.05) {
  cat("Conclusion: There IS a significant linear relationship between spend and retained impressions.\n")
} else {
  cat("Conclusion: No significant linear relationship at α = 0.05.\n")
}

# (c) Bands notice
cat("\n=== Part (c): 95% Confidence & Prediction Bands ===\n")
cat("Bands are shown in HW01_plots.pdf — Page 1: regression line only; Page 2: 95% CI (red) & 95% PI (blue).\n")

# (d) Intervals for MCI (amount = 26.9)
cat("\n=== Part (d): 95% CI & PI for MCI (amount = 26.9) ===\n")
MCI_amt <- 26.9
MCI_CI <- predict(model, newdata = data.frame(amount = MCI_amt),
                  interval = "confidence", level = 0.95)
MCI_PI <- predict(model, newdata = data.frame(amount = MCI_amt),
                  interval = "prediction", level = 0.95)
cat(sprintf("Mean prediction at MCI (fit) = %.4f\n", MCI_CI[1, "fit"]))
cat(sprintf("95%% CI for mean: [%.4f, %.4f]\n", MCI_CI[1, "lwr"], MCI_CI[1, "upr"]))
cat(sprintf("95%% PI for individual: [%.4f, %.4f]\n", MCI_PI[1, "lwr"], MCI_PI[1, "upr"]))

# =========================================================
# REQUIRED FILE #2: HW01_plots.pdf
# =========================================================
# Prepare bands
newdf    <- data.frame(amount = seq(min(df$amount), max(df$amount), by = 0.5))
confband <- predict(model, newdata = newdf, interval = "confidence",  level = 0.95)
predband <- predict(model, newdata = newdf, interval = "prediction", level = 0.95)

pdf("HW01_plots.pdf")

# Page 1: Scatter + regression line
plot(ret_impress ~ amount, data = df,
     pch = 21, bg = "black", col = "black",
     xlab = "Amount Spent (millions)",
     ylab = "Retained Impressions per Week (millions)",
     main = "Regression Line")
abline(model, lwd = 2)

# Page 2: Scatter + regression + 95% CI (red) + 95% PI (blue)
plot(ret_impress ~ amount, data = df,
     pch = 21, bg = "black", col = "black",
     xlab = "Amount Spent (millions)",
     ylab = "Retained Impressions per Week (millions)",
     main = "Regression with 95% CI (red) & 95% PI (blue)")
abline(model, lwd = 2)
lines(newdf$amount, confband[, "lwr"], col = "red",  lty = 2, lwd = 1.5)
lines(newdf$amount, confband[, "upr"], col = "red",  lty = 2, lwd = 1.5)
lines(newdf$amount, predband[, "lwr"], col = "blue", lty = 2, lwd = 1.5)
lines(newdf$amount, predband[, "upr"], col = "blue", lty = 2, lwd = 1.5)
legend("topleft",
       legend = c("Data", "Regression", "95% CI", "95% PI"),
       pch = c(21, NA, NA, NA), pt.bg = "black",
       lty = c(NA, 1, 2, 2), lwd = c(NA, 2, 1.5, 1.5),
       col = c("black", "black", "red", "blue"), bty = "n")

dev.off()
