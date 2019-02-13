
## Let's load the data! The data can be downloaded from the Elementary Statistical Learning packages
#install.packages("ElemStatLearn") ### note: if you have not installed this packages, you must remove the # at the front of this line
library(ElemStatLearn)

data(prostate)
names(prostate)


## Since we do not have any hyperparamters to tune, let's split the data into trainig and test sets!

train <- subset(prostate,train==TRUE)[,1:9]
test <- subset(prostate,train==FALSE)[,1:9]

nrow(train)
nrow(test)

## visualize the data

pairs( prostate[,1:9], col="blue" )

## Let's fit our model using the training data

fit1 <- lm(lpsa ~ ., data = train)
summary(fit1)
# lcavol and lweight are highly significant


# Compute 95% confidence interval
confint(fit1,level=0.95)


### How well does our model do when we generalize it to an independent test set?

# Prediction
prediction1 <- predict(fit1, newdata = test[,1:8])
predict(fit1, newdata = test[,1:8], interval = "confidence", level=0.95)


# Calculate the training error
mean(fit1$residuals^2)
# Calculate the prediction error
mean((prediction1-test[,9])^2)

## let's use cross-validation!
#install.packages("glmnet")
library(glmnet)  
# Perform 5-fold cross-validation to select lambda ---------------------------
lambdas_to_try <- 10^seq(-3, 5, length.out = 100)
# Setting alpha = 0 implements ridge regression
ridge_cv <- cv.glmnet(as.matrix(train[,1:8]), as.vector(train[,9]), alpha = 0, lambda = lambdas_to_try,
                      standardize = TRUE, nfolds = 5)  #alpha=0 means ridge regression
# Plot cross-validation results
plot(ridge_cv)

# Best cross-validated lambda
lambda_cv <- ridge_cv$lambda.min
lambda_cv
# Fit final model, get its sum of squared residuals, multiple R-squared, and training error
model_cv <- glmnet(as.matrix(train[,1:8]), as.vector(train[,9]), alpha = 0, lambda = lambda_cv, standardize = TRUE)
coef(model_cv)


#since lambda close to 0 should be close to OLS estimates

y_hat_cv <- predict(model_cv, as.matrix(train[,1:8]))
ssr_cv <- t(as.vector(train[,9]) - y_hat_cv) %*% (as.vector(train[,9]) - y_hat_cv)
rsq_ridge_cv <- cor(as.vector(train[,9]), y_hat_cv)^2


mean(ssr_cv)
rsq_ridge_cv

# Get test error
y_hat_cv <- predict(model_cv, newx=as.matrix(test[,1:8]), type="response")
#ssr_cv <- t(as.vector(test[,9]) - y_hat_cv) %*% (as.vector(test[,9]) - y_hat_cv)
rsq_ridge_cv <- cor(as.vector(test[,9]), y_hat_cv)^2

prediction.errors=mean((y_hat_cv-test[,9])^2) #mean squared error
prediction.errors  ## lower than before!
ssr_cv
#rsq_ridge_cv

# Perform 5-fold cross-validation to select lambda ---------------------------
lambdas_to_try <- 10^seq(-3, 5, length.out = 100)
# Setting alpha = 0 implements ridge regression
lasso_cv <- cv.glmnet(as.matrix(train[,1:8]), as.vector(train[,9]), alpha = 1, lambda = lambdas_to_try,
                      standardize = TRUE, nfolds = 5)  #alpha=0 means ridge regression
# Plot cross-validation results
plot(lasso_cv)

# Best cross-validated lambda
lambda_cv <- lasso_cv$lambda.min
lambda_cv
# Fit final model, get its sum of squared residuals and multiple R-squared
model_cv <- glmnet(as.matrix(train[,1:8]), as.vector(train[,9]), alpha = 1, lambda = lambda_cv, standardize = TRUE)

coef(model_cv)


y_hat_cv <- predict(model_cv, as.matrix(test[,1:8]))
ssr_cv <- t(as.vector(test[,9]) - y_hat_cv) %*% (as.vector(test[,9]) - y_hat_cv)
rsq_lasso_cv <- cor(as.vector(test[,9]), y_hat_cv)^2


prediction.errors=mean((y_hat_cv-test[,9])^2) 
prediction.errors  ## lower than before!
ssr_cv
#rsq_lasso_cv

## Let's look at how coefficients shrink
res <- glmnet(as.matrix(train[,1:8]), as.vector(train[,9]), alpha = 1, lambda = lambdas_to_try, standardize = FALSE)
plot(res, xvar = "lambda", col=1:8)
legend("bottomright", lwd = 1, col = 1:8, legend = colnames(as.matrix(train[,1:8])), cex = .7)


