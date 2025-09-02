# Load necessary library
library(faraway)

# Load the pima dataset
data(pima)

# Create a data frame for observations where test = 0
pima_test0 <- subset(pima, test == 0)

# Create a data frame for observations where test = 1
pima_test1 <- subset(pima, test == 1)

# Compute the mean of the diabetes variable for test = 0 and test = 1 using tapply()
diabetes_means_test <- tapply(pima$diabetes, pima$test, mean, na.rm = TRUE)

# Print results
cat("Mean of diabetes variable for test = 0:", diabetes_means_test["0"], "\n")
cat("Mean of diabetes variable for test = 1:", diabetes_means_test["1"], "\n")
