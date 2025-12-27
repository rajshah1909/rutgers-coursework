# ============================================================
# HW08 — Problem 7.26 
# Forward & Backward Order Determination
# Name: RAJ SHAH
# Date: 10/27/2025
# ============================================================

# Set working directory
setwd("C:/Users/RAJ RUTGERS/Desktop/Regression Methods/HW08")

# Load required libraries
library(olsrr)
library(car)

# Load dataset
p7 <- read.csv("p7_26.csv", stringsAsFactors = FALSE)

# Start writing all outputs to text file
sink("HW08_output.txt")

cat("============================================================\n")
cat("HW08 — Problem 7.26\n")
cat("Name: RAJ SHAH\n")
cat("============================================================\n\n")

# ============================================================
# MODEL DEFINITIONS
# ============================================================

# Define polynomial models
m_linear    <- lm(y ~ x, data = p7)
m_quadratic <- lm(y ~ x + I(x^2), data = p7)
m_cubic     <- lm(y ~ x + I(x^2) + I(x^3), data = p7)

# ============================================================
# (1) FORWARD SELECTION
# ============================================================

cat("(1) Forward Selection Results\n")
cat("------------------------------------------------------------\n\n")

forward_result <- ols_step_forward_p(m_linear, scope = ~ x + I(x^2) + I(x^3))
print(forward_result)

cat("\nInterpretation (Forward):\n")
cat("• Forward selection starts from the simplest model.\n")
cat("• Variables are added if significant (p < 0.05).\n")
cat("• The final selection indicates the preferred model order.\n\n")

# ============================================================
# (2) BACKWARD ELIMINATION
# ============================================================

cat("============================================================\n")
cat("(2) Backward Elimination Results\n")
cat("============================================================\n\n")

backward_result <- ols_step_backward_p(m_cubic)
print(backward_result)

cat("\nInterpretation (Backward):\n")
cat("• Backward elimination begins with the full cubic model.\n")
cat("• Terms are removed if non-significant (p > 0.05).\n")
cat("• The remaining terms indicate best model complexity.\n\n")

# ============================================================
# END OF OUTPUT
# ============================================================
sink()

# Final confirmation message
cat("\n✅ All HW08 results successfully saved to:\n")
cat("C:/Users/RAJ RUTGERS/Desktop/Regression Methods/HW08\n")