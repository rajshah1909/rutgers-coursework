# Load necessary library
library(faraway)

# Load the pima dataset
data(pima)

# Get summary statistics for BMI to determine quartile cutoffs
bmi_summary <- summary(pima$bmi)

# Extract the first (Q1) and third (Q3) quartile cutoffs
Q1_cutoff <- bmi_summary[2]  # 1st quartile
Q3_cutoff <- bmi_summary[5]  # 3rd quartile

# Categorize BMI into quartiles
pima$bmi_quartile <- cut(pima$bmi, 
                          breaks = c(-Inf, Q1_cutoff, Q3_cutoff, Inf),
                          labels = c("Q1", "Q2-Q3", "Q4"))

# Compute the mean of the diabetes variable for Q1 and Q4 using tapply()
diabetes_means <- tapply(pima$diabetes, pima$bmi_quartile, mean, na.rm = TRUE)

# Print results
cat("Mean of diabetes variable for Q1 (BMI in 1st quartile):", diabetes_means["Q1"], "\n")
cat("Mean of diabetes variable for Q4 (BMI in 4th quartile):", diabetes_means["Q4"], "\n")
