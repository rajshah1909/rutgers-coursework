# HW07 Solution

# Set seed and generate samples
set.seed(12345)
sample1 <- rnorm(10, mean=4, sd=1)
sample2 <- rnorm(10, mean=4.4, sd=1)

# a. Generate normal Q-Q plots for each sample
par(mfrow=c(1,2))  # Create a 1x2 plotting area
qqnorm(sample1, main="Q-Q Plot for Sample 1")
qqline(sample1)
qqnorm(sample2, main="Q-Q Plot for Sample 2")
qqline(sample2)

# Print the samples to verify our data
print("Sample 1:")
print(sample1)
print("Sample 2:")
print(sample2)

# Calculate means for verification
mean1 <- mean(sample1)
mean2 <- mean(sample2)
print(paste("Mean of sample 1:", mean1))
print(paste("Mean of sample 2:", mean2))

# b. Perform two-sample t-test
t_test_result <- t.test(sample1, sample2, var.equal=TRUE)
print("Two-sample t-test result:")
print(t_test_result)

# c. Perform permutation test
# First, install and load the EnvStats package if not already installed
# install.packages("EnvStats")
library(EnvStats)

# Calculate number of permutations
num_perms <- choose(20, 10)
print(paste("Number of permutations:", num_perms))

# Perform permutation test
perm_test_result <- twoSamplePermutationTestLocation(
  x=sample1, 
  y=sample2, 
  fcn="mean", 
  alternative="two.sided", 
  seed=123,
  n.permutations=num_perms
)

print("Permutation test result:")
print(perm_test_result)