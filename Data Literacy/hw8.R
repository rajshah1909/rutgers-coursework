# HW08 Solution: Comparing Uniform Distributions

# Set seed and generate samples
set.seed(12345)
sample1 <- runif(9, 0, 1)
sample2 <- runif(9, 0, 1.4)

# Print the samples to verify
cat("Sample 1 (Uniform[0,1]):\n")
print(sample1)
cat("\nSample 2 (Uniform[0,1.4]):\n")
print(sample2)

# Calculate and print the means
mean1 <- mean(sample1)
mean2 <- mean(sample2)
cat("\nMean of sample 1:", mean1)
cat("\nMean of sample 2:", mean2)
cat("\nDifference in means:", mean1 - mean2, "\n")

# a. Generate normal Q-Q plot for each sample
par(mfrow=c(1,2))  # Create a 1x2 plotting area
qqnorm(sample1, main="Q-Q Plot for Uniform[0,1]")
qqline(sample1)
qqnorm(sample2, main="Q-Q Plot for Uniform[0,1.4]")
qqline(sample2)

# b. Perform two-sample t-test
t_test_result <- t.test(sample1, sample2, var.equal=TRUE)
cat("\nb. Two-sample t-test result:\n")
print(t_test_result)

# c. Perform permutation test
# Load the EnvStats package
library(EnvStats)

# Calculate number of permutations
num_perms <- choose(18, 9)
cat("\nNumber of permutations (choose(18,9)):", num_perms, "\n")

# Perform permutation test
perm_test_result <- twoSamplePermutationTestLocation(
  x=sample1, 
  y=sample2, 
  fcn="mean", 
  alternative="two.sided", 
  seed=123,
  n.permutations=num_perms
)

cat("\nc. Permutation test result:\n")
print(perm_test_result)

# Summary of results
cat("\nSummary of Results:\n")
cat("t-test p-value:", t_test_result$p.value, "\n")
cat("Permutation test p-value:", perm_test_result$p.value, "\n")