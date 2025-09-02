# hw1.R

# Create a character vector with 8 values
c1 <- c("A", "B", "C", "D", "E", "F", "G", "H")

# Create two numeric vectors with 8 values each
v1 <- c(10, 20, 30, 40, 50, 60, 70, 80)
v2 <- c(5, 15, 25, 35, 45, 55, 65, 75)

# Use cbind() to create a data frame
hw1 <- as.data.frame(cbind(c1, v1, v2))

# Convert v1 and v2 to numeric (since cbind() may convert them to characters)
hw1$v1 <- as.numeric(hw1$v1)
hw1$v2 <- as.numeric(hw1$v2)

# Add a 4th variable v3 that adds v1 and v2
hw1$v3 <- hw1$v1 + hw1$v2

# Find the minimum of v3
min_v3 <- min(hw1$v3)

# Find the maximum of v3
max_v3 <- max(hw1$v3)

# Print the data frame and results
print(hw1)
cat("Minimum of v3:", min_v3, "\n")
cat("Maximum of v3:", max_v3, "\n")
