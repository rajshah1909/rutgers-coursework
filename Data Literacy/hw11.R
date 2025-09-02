
#HW11
library(rpart)
library(rpart.plot)
library(caret)
set.seed(12345)

#change this next line to point to the location of bankloan.csv
setwd("C:/Users/parth/Downloads")
df <- read.csv("bankloan.csv")

#drop unused columns from the data (ID and Zip Code)
df <- df[-c(1, 5)]

tree <- rpart(df$Personal.Loan  ~., data = df, method = 'class',control=rpart.control(cv=10,cp=.0001))
tree$cptable

#identify best cp value to use
(best <- tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"])
#produce a pruned tree based using the best cp value
pruned_tree <- prune(tree, cp=0.010)
print(pruned_tree)
rpart.plot(pruned_tree)

pred <-predict(pruned_tree,data=df,type = 'class')

#Generate Confusion Matrix
table_rpart = table(pred,df$Personal.Loan)
caret::confusionMatrix(table_rpart,positive="1")
##---------------------------------------------##

 