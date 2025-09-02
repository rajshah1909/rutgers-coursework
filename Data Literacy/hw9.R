
install.packages(c("e1071", "caret", "dplyr", "ggplot2"))

# Load required libraries
library(e1071)   # For naive Bayes
library(caret)   # For confusion matrix
library(dplyr)   # For data manipulation
library(ggplot2) # For visualization

# Set working directory to where your files are located
setwd("C:/Users/parth/Downloads")

# Step 1: Load the datasets
training_data <- read.csv("training_email_advertising.csv", stringsAsFactors = TRUE)
test_data <- read.csv("test_email_advertising.csv", stringsAsFactors = TRUE)

# Step 2: Explore the data
str(training_data)
summary(training_data)

# Make sure the response variable is a factor
training_data$none_open_buy <- as.factor(training_data$none_open_buy)
test_data$none_open_buy <- as.factor(test_data$none_open_buy)

# Convert other variables to factors as needed
training_data$purchased_previously <- as.factor(training_data$purchased_previously)
training_data$opened_previously <- as.factor(training_data$opened_previously)
training_data$test_var <- as.factor(training_data$test_var)

test_data$purchased_previously <- as.factor(test_data$purchased_previously)
test_data$opened_previously <- as.factor(test_data$opened_previously)
test_data$test_var <- as.factor(test_data$test_var)

# Step 3: Build the Naive Bayes model
# Using all features
nb_model <- naiveBayes(none_open_buy ~ purchased_previously + opened_previously + 
                      sales_12mo + test_var, 
                      data = training_data)

# Print model details
print(nb_model)

# Step 4: Make predictions
# Predictions on training data
train_pred <- predict(nb_model, training_data)

# Predictions on test data
test_pred <- predict(nb_model, test_data)

# Step 5: Generate confusion matrices
# 1. Confusion matrix for training data
train_confusion <- confusionMatrix(train_pred, training_data$none_open_buy)
print("Confusion Matrix for Training Data:")
print(train_confusion)

# 2. Confusion matrix for test data
test_confusion <- confusionMatrix(test_pred, test_data$none_open_buy)
print("Confusion Matrix for Test Data:")
print(test_confusion)

# Step 6: Visualize confusion matrices
# Function to plot confusion matrix
plot_confusion_matrix <- function(cm, title) {
  cm_table <- as.data.frame(cm$table)
  
  # Convert to percentages within each actual class
  cm_table_pct <- cm_table %>%
    group_by(Reference) %>%
    mutate(Percentage = Freq / sum(Freq) * 100)
  
  # Plot
  ggplot(cm_table, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), color = "black") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(title = title,
         x = "Actual Class",
         y = "Predicted Class") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
}

# Plot training confusion matrix
train_plot <- plot_confusion_matrix(train_confusion, "Training Data Confusion Matrix")
print(train_plot)

# Plot test confusion matrix
test_plot <- plot_confusion_matrix(test_confusion, "Test Data Confusion Matrix")
print(test_plot)

# Save plots if needed
# ggsave("training_confusion_matrix.png", train_plot, width = 7, height = 5)
# ggsave("test_confusion_matrix.png", test_plot, width = 7, height = 5)

# Additional analysis: detailed performance metrics
cat("\n\nDetailed Performance Metrics for Training Data:\n")
print(train_confusion$overall)
print(train_confusion$byClass)

cat("\n\nDetailed Performance Metrics for Test Data:\n")
print(test_confusion$overall)
print(test_confusion$byClass)

# Summary of findings
cat("\n\nSummary of Naive Bayes Classification:\n")
cat("Training Accuracy:", round(train_confusion$overall["Accuracy"] * 100, 2), "%\n")
cat("Test Accuracy:", round(test_confusion$overall["Accuracy"] * 100, 2), "%\n\n")

# Compare class-specific performances
performance_comparison <- data.frame(
  Class = c("Class 1", "Class 2", "Class 3"),
  Training_Sensitivity = c(
    train_confusion$byClass[1, "Sensitivity"],
    train_confusion$byClass[2, "Sensitivity"],
    train_confusion$byClass[3, "Sensitivity"]),
  Test_Sensitivity = c(
    test_confusion$byClass[1, "Sensitivity"],
    test_confusion$byClass[2, "Sensitivity"],
    test_confusion$byClass[3, "Sensitivity"]),
  Training_Precision = c(
    train_confusion$byClass[1, "Pos Pred Value"],
    train_confusion$byClass[2, "Pos Pred Value"],
    train_confusion$byClass[3, "Pos Pred Value"]),
  Test_Precision = c(
    test_confusion$byClass[1, "Pos Pred Value"],
    test_confusion$byClass[2, "Pos Pred Value"],
    test_confusion$byClass[3, "Pos Pred Value"])
)

print(performance_comparison)