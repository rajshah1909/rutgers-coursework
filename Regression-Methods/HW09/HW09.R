# ============================================================
# HW09.R
# ============================================================

# Load required libraries
library(faraway)
library(olsrr)

# Set working directory and load dataset
setwd("C:/Users/RAJ RUTGERS/Desktop/Regression Methods/HW09")
ridge <- read.csv("ridge_data.csv")

# Start saving all output to file
sink("HW09_output.txt")

cat("============================================================\n")
cat("RAJ SHAH\n")
cat("HW09 — Model Selection using ridge_data.csv\n")
cat("============================================================\n\n")

# Define full model
full_model <- lm(y ~ x1 + x2 + x3 + x4 + x5 + x8, data = ridge)

# ============================================================
# (a) Forward Stepwise Selection
# ============================================================
cat("(a) Forward Stepwise Selection\n")
cat("------------------------------------------------------------\n")
forward_model <- ols_step_forward_p(full_model, details = TRUE)
print(forward_model)

cat("\nInterpretation:\n")
cat("The forward stepwise method starts with no predictors and adds them one by one.\n")
cat("The final chosen model includes the predictors that significantly improve the model fit.\n\n")

# ============================================================
# (b) Backward Stepwise Elimination
# ============================================================
cat("(b) Backward Stepwise Elimination\n")
cat("------------------------------------------------------------\n")
backward_model <- ols_step_backward_p(full_model, details = TRUE)
print(backward_model)

cat("\nInterpretation:\n")
cat("The backward elimination method begins with all predictors and removes the least significant ones.\n")
cat("The final chosen model retains only the variables that meaningfully contribute to predicting y.\n\n")

# ============================================================
# (c) Bi-directional Stepwise Selection
# ============================================================
cat("(c) Bi-directional Stepwise Selection\n")
cat("------------------------------------------------------------\n")
both_model <- ols_step_both_p(full_model, details = TRUE)
print(both_model)

cat("\nInterpretation:\n")
cat("The bi-directional stepwise selection allows both adding and removing predictors in each step.\n")
cat("The resulting model represents the best combination found by balancing significance in both directions.\n\n")

# ============================================================
# (d) All Possible Regressions using x1, x2, x3, x4, x5, x8
# ============================================================
cat("(d) All Possible Regressions\n")
cat("------------------------------------------------------------\n")
all_models <- ols_step_all_possible(full_model)
print(all_models)

# Save plot for visual output
png("HW09_AllPossible.png", width = 900, height = 700)
plot(all_models)
dev.off()

# Choose best model by Adjusted R²
best_model <- all_models$models[which.max(all_models$adjr), ]
cat("\nBest Model based on Adjusted R²:\n")
print(best_model)

cat("\nInterpretation:\n")
cat("Among all possible regressions, the model with the highest Adjusted R² is chosen.\n")
cat("The best model includes predictors x1 and x4, which together achieve the highest Adjusted R² (0.7605).\n")
cat("This model provides an excellent fit while maintaining simplicity (parsimony) by avoiding unnecessary predictors.\n")

cat("\n============================================================\n")
cat("Output successfully saved to HW09_output.txt\n")
cat("============================================================\n")

sink()