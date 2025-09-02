# Load the data
# Note: You'll need to adjust the file path to where you've downloaded the file
data <- read.csv("hw5_data.csv")

# Part a: Generate a 95% confidence interval on the mean
mean_diaBP <- mean(data$diaBP)
sd_diaBP <- sd(data$diaBP)
n <- length(data$diaBP)
error <- qt(0.975, df = n-1) * sd_diaBP / sqrt(n)
ci_lower <- mean_diaBP - error
ci_upper <- mean_diaBP + error

# Part b: Calculate the margin of error
margin_of_error <- error

# Part c: Calculate the sample size needed for margin of error < 5
# Using the formula: n = (z*s/E)^2
# where z is the critical value, s is the sample standard deviation, and E is the desired margin of error
desired_error <- 5
z_value <- qnorm(0.975)  # z-value for 95% confidence
required_n <- ceiling((z_value * sd_diaBP / desired_error)^2)

# Print the results
cat("Part a: 95% Confidence Interval\n")
cat("Sample Mean:", mean_diaBP, "\n")
cat("95% CI: [", ci_lower, ", ", ci_upper, "]\n\n")

cat("Part b: Margin of Error\n")
cat("Margin of Error:", margin_of_error, "\n\n")

cat("Part c: Required Sample Size\n")
cat("Sample size needed for margin of error < 5:", required_n, "\n")

