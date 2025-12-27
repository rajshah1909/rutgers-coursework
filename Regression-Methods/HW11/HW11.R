#HW11
#Use the code below to perform Ridge and LASSO analyses on the first 40 predictors in the meatspec dataset. 
#Use the glmnet package. You do not need to build training and test data sets for this exercise.
#Use 10-fold Cross Validation to pick the optimal lambdas.
#Submit all output, and any graphs that are generated.
 
library(faraway)
library(glmnet)

#save graphics output in pdf - saves graph(s) in working directory
getwd() #lists the working directory where you can find the file HW11_out.pdf
pdf (file="HW11_out.pdf" )

data(meatspec,package="faraway")  #the dataset meatspec is found in the faraway package
meatspec=meatspec[,-(41:100)]
attach(meatspec)

set.seed(15342)

# Getting the independent variables
x_var <- as.matrix(meatspec[,(1:40)]) #includes the 40 variables V1 – V40
print(head(x_var,1L))
 
# Getting the dependent variable
y_var <- meatspec$fat

#continue with program to perform Ridge and LASSO.
# Using glmnet function to build the ridge regression in r
fit <- glmnet(x_var, y_var, alpha = 0)
# Checking the model
summary(fit)

# Using cross validation glmnet
ridge_cv <- cv.glmnet(x_var, y_var, alpha = 0)    #default number of folds for CV is 10
# Best lambda value
plot(ridge_cv)
(best_lambda <- ridge_cv$lambda.min)

best_fit <- ridge_cv$glmnet.fit
print(head(best_fit,1L))

# Rebuilding the model with optimal lambda value
best_ridge <- glmnet(x_var, y_var, alpha = 0, lambda =  best_lambda)
print(coef(best_ridge))     #obtain the Ridge Regression coefficients at the lambda=CV best lambda 
plot(best_fit, xvar = "lambda", label = TRUE)

#repeat similar code but use alpha=1 for LASSO regression

# Using glmnet function to build the lasso regression in r
fit <- glmnet(x_var, y_var, alpha = 1)
# Checking the model
summary(fit)

# Using cross validation glmnet
lasso_cv <- cv.glmnet(x_var, y_var, alpha = 1)   #default number of folds for CV is 10
# Best lambda value
plot(lasso_cv)
(best_lambda <- lasso_cv$lambda.min)
best_fit <- lasso_cv$glmnet.fit
best_fit

# Rebuilding the model with optimal lambda value
best_lasso <- glmnet(x_var, y_var, alpha = 1, lambda = best_lambda)
sum(coef(best_lasso) != 0)
coef(best_lasso)               #obtain the LASSO coefficients at the lambda=CV best lambda  
#note the selected variables tend to be grouped. 

lmod <- lm(fat ~ ., meatspec)  #obtain coefficents of OLS model
print(coef(lmod))
plot(best_fit, xvar = "lambda", label = TRUE)

##-------------------------------##
dev.off() #closes pdf file
