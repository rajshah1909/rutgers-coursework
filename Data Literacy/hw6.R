# Load the data
# Note: You'll need to adjust the file path to where you've downloaded the file
data <- read.csv("hw6_data.csv")

# Part a: Generate a 95% confidence interval on the proportion of females
# First, calculate the proportion of females in the sample
female_count <- sum(data$isFemale)
n <- length(data$isFemale)
p_hat <- female_count / n

# Calculate standard error of the proportion
se <- sqrt((p_hat * (1 - p_hat)) / n)

# Calculate the 95% confidence interval
z <- qnorm(0.975)  # z-value for 95% confidence
ci_lower <- p_hat - z * se
ci_upper <- p_hat + z * se

# Part b: Calculate the margin of error
margin_of_error <- z * se

# Part c: Calculate the sample size needed for margin of error < 0.05 (5%)
# Using the formula: n = (z^2 * p_hat * (1-p_hat)) / E^2
# where z is the critical value, p_hat is the sample proportion, and E is the desired margin of error
desired_error <- 0.05
required_n <- ceiling((z^2 * p_hat * (1 - p_hat)) / (desired_error^2))

# Print the results
cat("Part a: 95% Confidence Interval for Proportion of Females\n")
cat("Sample Size:", n, "\n")
cat("Number of Females:", female_count, "\n")
cat("Sample Proportion:", p_hat, "\n")
cat("95% CI: [", ci_lower, ", ", ci_upper, "]\n\n")

cat("Part b: Margin of Error\n")
cat("Margin of Error:", margin_of_error, "\n\n")

cat("Part c: Required Sample Size\n")
cat("Sample size needed for margin of error < 0.05:", required_n, "\n")

# Alternative calculation using prop.test function for verification
prop_test_result <- prop.test(female_count, n, conf.level = 0.95, correct = FALSE)
cat("\nVerification using prop.test function:\n")
print(prop_test_result$conf.int)