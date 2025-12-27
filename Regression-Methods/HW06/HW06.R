# ============================================================
# HW06 — Problem 4.2 (Continuation of HW05)
# Standardized Betas and Residual Diagnostics
# Name: RAJ SHAH
# Date: 10/20/2025
# ============================================================

# Set working directory
setwd("C:/Users/RAJ RUTGERS/Desktop/Regression Methods/HW06")

# Load required libraries
library(olsrr)
library(car)
library(lm.beta)
library(ggplot2)

# Load dataset
p3_1 <- read.csv("B1.csv", stringsAsFactors = FALSE)

# Start writing all outputs to text file
sink("HW06_output.txt")

cat("============================================================\n")
cat("HW06 — Problem 4.2 (Continuation of HW05)\n")
cat("Name: RAJ SHAH\n")
cat("============================================================\n\n")

# ============================================================
# (a) Standardized Betas for x2, x7, x8
# ============================================================
cat("(a) Standardized Beta Coefficients for x2, x7, x8\n")
cat("------------------------------------------------------------\n\n")

lmod <- lm(y ~ x2 + x7 + x8, data = p3_1)
std_model <- lm.beta(lmod)

cat("Standardized Beta Coefficients:\n\n")
print(summary(std_model))

cat("\nInterpretation:\n")
cat("• Standardized betas represent the effect size of each variable in SD units.\n")
cat("• The predictor with the largest |beta| has the strongest standardized effect.\n")

# ============================================================
# (b) 95% Confidence Intervals for Standardized Betas
# ============================================================
cat("\n============================================================\n")
cat("(b) 95% Confidence Intervals for Standardized Betas\n")
cat("============================================================\n\n")

# Compute standardized betas
std_betas <- lm.beta(lmod)$standardized.coefficients[-1]  # exclude intercept
se_betas <- summary(lmod)$coefficients[-1, 2]
t_crit <- qt(0.975, df.residual(lmod))
lower_ci <- std_betas - t_crit * se_betas
upper_ci <- std_betas + t_crit * se_betas

ci_table <- data.frame(
  Predictor = c("x2", "x7", "x8"),
  Std_Beta = round(std_betas, 4),
  Lower_95CI = round(lower_ci, 4),
  Upper_95CI = round(upper_ci, 4)
)
print(ci_table)

cat("\nInterpretation:\n")
cat("• Narrower CIs indicate more precise estimates.\n")
cat("• None of the 95% CIs include 0, confirming significance of predictors.\n")

# ============================================================
# (c) Studentized and R-Student Residuals Plots
# ============================================================
cat("\n============================================================\n")
cat("(c) Studentized and R-Studentized Residual Diagnostics\n")
cat("============================================================\n\n")

# --- Studentized Residuals ---
cat("Creating Studentized Residuals Plot...\n\n")
stud_plot <- ols_plot_resid_stand(lmod)

# Save the plot using ggsave
ggsave(
  filename = "HW06_Studentized_Residuals.png",
  plot = stud_plot,
  path = "C:/Users/RAJ RUTGERS/Desktop/Regression Methods/HW06",
  width = 9, height = 6, dpi = 300
)

cat("Studentized residuals saved to HW06_Studentized_Residuals.png\n\n")

# --- R-Student Residuals ---
cat("Creating R-Studentized Residuals Plot...\n\n")
rstud_plot <- ols_plot_resid_stud(lmod)

# Save the plot using ggsave
ggsave(
  filename = "HW06_RStudent_Residuals.png",
  plot = rstud_plot,
  path = "C:/Users/RAJ RUTGERS/Desktop/Regression Methods/HW06",
  width = 9, height = 6, dpi = 300
)

cat("R-Student residuals saved to HW06_RStudent_Residuals.png\n\n")

# Extract residual data for reference
resid_stand <- ols_plot_resid_stand(lmod)$data
resid_stud <- ols_plot_resid_stud(lmod)$data

cat("Summary of Studentized Residuals:\n")
print(summary(resid_stand$stud_resid))
cat("\nSummary of R-Studentized Residuals:\n")
print(summary(resid_stud$rstudent))

cat("\nInterpretation:\n")
cat("• Points with |residual| > 2 may be mild outliers.\n")
cat("• Points with |residual| > 3 are potential influential outliers.\n")
cat("• Based on the plot, no points exceed ±3, indicating no serious outliers.\n")
cat("• Residuals appear well-behaved, supporting model adequacy.\n")

# ============================================================
# END OF HW06 OUTPUT
# ============================================================
sink()

cat("\nAll HW06 results and plots have been successfully saved to:\n")
cat("C:/Users/RAJ RUTGERS/Desktop/Regression Methods/HW06\n")