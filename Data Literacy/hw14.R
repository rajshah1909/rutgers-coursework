# HW14: Salk Vaccine Trials Analysis
# Testing for independence between Vaccination Status and Polio Status

# Create the contingency table
salk_data <- matrix(c(
  200688, 24, 33,     # Vaccinated
  201087, 27, 115     # Vaccinated with Placebo
), nrow = 2, byrow = TRUE)

# Add row and column names
rownames(salk_data) <- c("Vaccinated", "Vaccinated with Placebo")
colnames(salk_data) <- c("Did not contract Polio", "Contracted Nonparalytic Polio", "Contracted Paralytic Polio")

# Display the contingency table
cat("Contingency Table:\n")
print(salk_data)
cat("\n")

# 1. Chi-Square Test for Independence
cat("1. Chi-Square Test for Independence\n")
cat("----------------------------------\n")
chi_sq_test <- chisq.test(salk_data)
print(chi_sq_test)
cat("\nChi-square test p-value:", chi_sq_test$p.value, "\n\n")

# 2. Permutation Test for Independence with 5000 permutations
cat("2. Permutation Test for Independence with 100 permutations\n")
cat("----------------------------------------------------------\n")

# Function to perform permutation test
perform_permutation_test <- function(data, n_permutations = 100) {
  # Original chi-square statistic
  observed_stat <- chisq.test(data)$statistic
  
  # Create a vector to store permutation statistics
  perm_stats <- numeric(n_permutations)
  
  # Flatten the table to a vector of categories
  n_total <- sum(data)
  all_values <- rep(1:ncol(data), colSums(data))
  
  # Row totals (to maintain in permutations)
  row_totals <- rowSums(data)
  
  # Perform permutations
  set.seed(42) # For reproducibility
  
  for (i in 1:n_permutations) {
    # Shuffle the categories
    shuffled_values <- sample(all_values)
    
    # Reconstruct the table with the same row totals
    perm_table <- matrix(0, nrow = nrow(data), ncol = ncol(data))
    
    # First row based on row total
    row1_indices <- sample(1:length(shuffled_values), row_totals[1])
    
    # Count occurrences of each category in row 1
    for (j in 1:ncol(data)) {
      perm_table[1, j] <- sum(shuffled_values[row1_indices] == j)
      perm_table[2, j] <- sum(data[, j]) - perm_table[1, j]
    }
    
    # Calculate chi-square statistic for this permutation
    perm_stats[i] <- chisq.test(perm_table)$statistic
  }
  
  # Calculate p-value (proportion of permutation statistics >= observed)
  p_value <- mean(perm_stats >= observed_stat)
  
  # Return results
  return(list(
    observed_stat = observed_stat,
    perm_stats = perm_stats,
    p_value = p_value
  ))
}

# Run the permutation test
perm_results <- perform_permutation_test(salk_data, 100)

# Output results
cat("Observed chi-square statistic:", perm_results$observed_stat, "\n")
cat("Permutation test p-value:", perm_results$p_value, "\n\n")

# Optional: Plot the distribution of permutation statistics
cat("Summary of permutation statistics:\n")
print(summary(perm_results$perm_stats))

# Calculate exact p-value for comparison
cat("\nExact p-value from chi-square distribution:", 
    pchisq(perm_results$observed_stat, df = (nrow(salk_data)-1)*(ncol(salk_data)-1), 
           lower.tail = FALSE), "\n")

# Conclusion
cat("\nConclusion:\n")
if (chi_sq_test$p.value < 0.05) {
  cat("Based on the chi-square test (p-value < 0.05), we reject the null hypothesis.\n")
  cat("There is sufficient evidence to conclude there is a dependence between\n")
  cat("Vaccination Status and Polio Status.\n")
} else {
  cat("Based on the chi-square test (p-value >= 0.05), we fail to reject the null hypothesis.\n")
  cat("There is not sufficient evidence to conclude there is a dependence between\n")
  cat("Vaccination Status and Polio Status.\n")
}

if (perm_results$p_value < 0.05) {
  cat("\nThe permutation test (p-value < 0.05) confirms this conclusion.\n")
} else if (perm_results$p_value >= 0.05 && chi_sq_test$p.value < 0.05) {
  cat("\nHowever, the permutation test (p-value >= 0.05) suggests a different conclusion.\n")
} else {
  cat("\nThe permutation test (p-value >= 0.05) confirms this conclusion.\n")
}