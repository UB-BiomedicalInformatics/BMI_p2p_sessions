
#We are going to focus on Classification, 
#predicting if a patient is diabetic or not

## load the data
#install.packages("mlbench")
#install.packages("corrplot")
#install.packages("caret")
#install.packages("randomForest")
#install.packages("e1071")
#install.packages("tree")
#install.packages("dplyr", dependencies = TRUE)
#library(dplyr)
library(mlbench)
library(corrplot)
library(caret)
library(randomForest)
library(tree)
library(e1071)


data("PimaIndiansDiabetes2", package = "mlbench")
?PimaIndiansDiabetes

## set seed for randomization
set.seed(123)

head(PimaIndiansDiabetes2)
pima<- PimaIndiansDiabetes2
## next we need to create a factor from our response variable
pima$diabetes <- as.factor(pima$diabetes)
nrow(pima)

sapply(pima, function(x) sum(is.na(x)))

# for now we will ignore the missing data
pima_nomiss<- na.omit(pima)
nrow(pima_nomiss)
# exploratory analysis of our data
#scatterplot
pairs(pima_nomiss, panel = panel.smooth)
#correlation matrix
corrplot(cor(pima_nomiss[, -9]), type = "lower", method = "number")

## split our data into train and test sets

# Set the fraction of the training data
training.fraction <- 0.7

# Train and Test Split using randomization
sample.size <- nrow(pima_nomiss)
index <- sample(1:sample.size)
n.train <- floor(sample.size*training.fraction)
training.data <- pima_nomiss[index[1:n.train],]
testing.data <- pima_nomiss[-index[1:n.train],]
dim(training.data)
dim(testing.data)

## decision tree of classifying to diabetes

# Training The Model
treemod <- tree(diabetes ~ ., data = training.data)

summary(treemod)



treemod # get a detailed text output.

plot(treemod)
text(treemod, pretty = 0)

### Lets test the model using our held-out test set.

tree_pred <- predict(treemod, newdata = testing.data, type = "class" )
confusionMatrix(tree_pred, testing.data$diabetes, positive="pos")

A<- confusionMatrix(tree_pred, testing.data$diabetes, positive="pos")

F1<- 2* (A$byClass['Sensitivity']* A$byClass['Pos Pred Value'])/(A$byClass['Sensitivity']+ A$byClass['Pos Pred Value'])
F1

# Set the fraction of the training data
training.fraction <- 0.8

# Split training data into train and validation
sample.size <- nrow(training.data)
index <- sample(1:sample.size)
n.train <- floor(sample.size*training.fraction)
training.data <- training.data[index[1:n.train],]
validation.data <- training.data[-index[1:n.train],]
dim(training.data)
dim(validation.data)

rf.pima_fin<-randomForest(diabetes ~., data=training.data,  mtry=3, ntree=1000, importance=T)
rf.pima_fin
importance(rf.pima_fin)
varImpPlot(rf.pima_fin, sort=T)

yhat = predict(rf.pima_fin, testing.data)
y = testing.data$diabetes
mean(y != yhat)

plot(rf.pima_fin)

## with cross validation
# split our data into train and test sets

# Set the fraction of the training data
training.fraction <- 0.7

# Train and Test Split using randomization
sample.size <- nrow(pima_nomiss)
index <- sample(1:sample.size)
n.train <- floor(sample.size*training.fraction)
training.data <- pima_nomiss[index[1:n.train],]
testing.data <- pima_nomiss[-index[1:n.train],]
dim(training.data)
dim(testing.data)

# Fit the model on the training set
model <- train(
  diabetes ~., data = training.data, method = "rf",
  trControl = trainControl("cv", number = 10),
  importance = TRUE
  )
# Best tuning parameter
model$bestTune
# Final model
model$finalModel
# Make predictions on the test data
predicted.classes <- predict(model,testing.data)
# Compute model accuracy rate
mean(predicted.classes == testing.data$diabetes)

#Compute error
mean(predicted.classes != testing.data$diabetes)

#install.packages("mice", dependencies=TRUE)
library(mice)

# Let's use decision trees (CART) to impute missing values
imp = mice(pima, meth = "cart", minbucket = 5)
imp1 = complete(imp)

nrow(imp1)

head(imp1)

###build random forest model with full dataset and cross validation

## with cross validation
# split our data into train and test sets

# Set the fraction of the training data
training.fraction <- 0.7

# Train and Test Split using randomization
sample.size <- nrow(imp1)
index <- sample(1:sample.size)
n.train <- floor(sample.size*training.fraction)
training.data <- imp1[index[1:n.train],]
testing.data <- imp1[-index[1:n.train],]
dim(training.data)
dim(testing.data)

# Fit the model on the training set
model <- train(
  diabetes ~., data = training.data, method = "rf",
  trControl = trainControl("cv", number = 10),
  importance = TRUE, ntree=500
  )
# Best tuning parameter
model$bestTune
# Final model
model$finalModel
importance(model$finalModel)
# Make predictions on the test data
predicted.classes <- predict(model,testing.data)
# Compute model accuracy rate
mean(predicted.classes == testing.data$diabetes)

#Compute error
mean(predicted.classes != testing.data$diabetes)



