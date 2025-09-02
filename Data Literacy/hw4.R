# Load necessary library
library(rpart)

# Load and explore the stagec dataset
help(stagec)

# Load the dataset
data(stagec, package = "rpart")


# (a) Generate a one-way table for the variable 'grade' using tapply()
cat("\nOne-way table for the 'grade' variable:\n")
grade_counts <- tapply(stagec$grade, stagec$grade, length)
print(grade_counts)

# Generate a two-way table for 'grade' vs 'eet' using tapply()
cat("\nTwo-way table for 'grade' vs 'eet':\n")
grade_eet_counts <- tapply(stagec$grade, list(stagec$grade, stagec$eet), length, default = 0)
print(grade_eet_counts)

# (b) Generate a subset of the data where pgstat = 1 (patients with progression)
stagec_subset <- subset(stagec, pgstat == 1)

# Provide a summary of the variable 'pgtime' in the subset using tapply()
cat("\nSummary of 'pgtime' variable for patients with progression (pgstat = 1):\n")
pgtime_summary <- tapply(stagec_subset$pgtime, stagec_subset$pgstat, summary)
print(pgtime_summary[[1]])  # Extract summary for pgstat = 1

# (c) Generate a one-way table for 'grade' in the subset using tapply()
cat("\nOne-way table for 'grade' in the subset (pgstat = 1):\n")
grade_counts_subset <- tapply(stagec_subset$grade, stagec_subset$grade, length)
print(grade_counts_subset)

# Generate a two-way table for 'grade' vs 'eet' in the subset using tapply()
cat("\nTwo-way table for 'grade' vs 'eet' in the subset (pgstat = 1):\n")
grade_eet_counts_subset <- tapply(stagec_subset$grade, list(stagec_subset$grade, stagec_subset$eet), length, default = 0)
print(grade_eet_counts_subset)
