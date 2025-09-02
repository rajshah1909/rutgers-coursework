# Part 1: Probability that sum of two dice is at least 4
cat("PART 1: Probability that sum of two dice is at least 4\n")
cat("Actual probability: 0.917\n\n")

# Set seed for reproducibility
set.seed(12345)

# Function to simulate rolling two dice and check if sum is at least 4
simulate_dice_sum_at_least_4 <- function(n_simulations) {
  # Generate n_simulations rolls of two dice
  die1 <- sample(1:6, n_simulations, replace = TRUE)
  die2 <- sample(1:6, n_simulations, replace = TRUE)
  
  # Calculate sum of the two dice
  sum_dice <- die1 + die2
  
  # Check how many sums are at least 4
  successes <- sum(sum_dice >= 4)
  
  # Calculate proportion of successes
  probability <- successes / n_simulations
  
  return(probability)
}

# Run simulations with different sample sizes
n_values <- c(5000, 25000, 100000)

for (n in n_values) {
  prob <- simulate_dice_sum_at_least_4(n)
  error <- abs(prob - 0.917)
  cat(sprintf("Simulations: %d, Estimated probability: %.5f, Error: %.5f\n", 
              n, prob, error))
}

cat("\n")

# Part 2: Probability that sum of two dice is at least 11
cat("PART 2: Probability that sum of two dice is at least 11\n")
cat("Actual probability: 0.083\n\n")

# Set seed for reproducibility
set.seed(13245)

# Function to simulate rolling two dice and check if sum is at least 11
simulate_dice_sum_at_least_11 <- function(n_simulations) {
  # Generate n_simulations rolls of two dice
  die1 <- sample(1:6, n_simulations, replace = TRUE)
  die2 <- sample(1:6, n_simulations, replace = TRUE)
  
  # Calculate sum of the two dice
  sum_dice <- die1 + die2
  
  # Check how many sums are at least 11
  successes <- sum(sum_dice >= 11)
  
  # Calculate proportion of successes
  probability <- successes / n_simulations
  
  return(probability)
}

# Run simulations with different sample sizes
n_values <- c(5000, 25000, 100000)

for (n in n_values) {
  prob <- simulate_dice_sum_at_least_11(n)
  error <- abs(prob - 0.083)
  cat(sprintf("Simulations: %d, Estimated probability: %.5f, Error: %.5f\n", 
              n, prob, error))
}