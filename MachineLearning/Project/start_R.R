##########################################################
##      
##      MIMIC Project R
##      Prepared by 
##      Topic: Predict mortality with high 
##      accuracy using logistic regression, decision tree, 
##      and random forest
##
##########################################################

## Set working directory ====

setwd("/Users/sarahmullin/Desktop/BMI507/MIMIC_project")

## Install Packages ====

#install.packages("tidyverse")   
#install.packages("dplyr")       
#install.packages("ggplot2")       
#install.packages("readxl")     
#install.packages("cluster")
#install.packages("onehot")
#install.packages("RPostgreSQL")

## Load required packages ====

library(tidyverse) 
#library(jsonlite)
#library(dplyr)
#library(dbplyr)
#library((RPostgreSQL)
library(dplyr)
library(icd)
library(lubridate)

#random forest packages
library(mlbench)
library(corrplot)
library(caret)
library(randomForest)
library(tree)
library(e1071)

#logistic regression packages
library(caTools)
#install.packages("ROCR")
library(ROCR)

## Load data from files: MIMIC-III ====

Patients_Admissions<-readRDS("Patients_Admissions.RDS")
Diagnoses<-readRDS("Diagnoses.RDS")
Procedures<-readRDS("Procedures.RDS")

head(Patients_Admissions)
head(Diagnoses)
head(Procedures)

# Create features

#Demographic features
X<- Patients_Admissions %>% dplyr::select(SUBJECT_ID, INSURANCE, MARITAL_STATUS, ETHNICITY, GENDER, EXPIRE_FLAG)
# create factors for prediction
X$INSURANCE<- as.factor(X$INSURANCE)
X$MARITAL_STATUS<- as.factor(X$MARITAL_STATUS)
X$ETHNICITY<- as.factor(X$ETHNICITY)
X$GENDER<- as.factor(X$GENDER)

# Create features from ICD codes

#ischaemic stroke
I_stroke<- Diagnoses %>% filter(ICD9_CODE=="436" | ICD9_CODE=="43310"  | ICD9_CODE=="43311" |
                                       ICD9_CODE=="43321"| ICD9_CODE=="43331"| ICD9_CODE=="43381"| 
                                       ICD9_CODE=="43391"| ICD9_CODE=="43400"| ICD9_CODE=="43401"|
                                      ICD9_CODE=="43411"| ICD9_CODE=="43491") %>% distinct(SUBJECT_ID) %>% 
  rowwise() %>% mutate(Ischaemic = 1)
X<- left_join(X, I_stroke) %>% mutate(Ischaemic = replace_na(Ischaemic, 0))
# Hemorrhagic Stroke
H_stroke<- Diagnoses %>% filter(ICD9_CODE=="430" | ICD9_CODE=="431") %>% distinct(SUBJECT_ID) %>% 
  rowwise() %>% mutate(Hemorrhagic = 1)
X<- left_join(X, H_stroke) %>% mutate(Hemorrhagic = replace_na(Hemorrhagic, 0))

#check to make sure you have cases that have these features!
table(X$Ischaemic)
table(X$Hemorrhagic)


## create features from CPT
  
# i want all of the surgery CPT codes
Procedures <- Procedures %>% filter(SECTIONHEADER=="Surgery")

# chest tube insertion

Chesttube<- Procedures %>% filter(CPT_CD=="32551") %>% distinct(SUBJECT_ID) %>% 
  rowwise() %>% mutate(Chesttube = 1)
X<- left_join(X, Chesttube) %>% mutate(Hemorrhagic = replace_na(Chesttube, 0))
table(X$Chesttube)

X$Chesttube<- as.factor(X$Chesttube)
X$Ischaemic<- as.factor(X$Ischaemic)
X$Hemorrhagic<- as.factor(X$Hemorrhagic)
#randomly split your dataset

## set seed for randomization
set.seed(123)

# Set the fraction of the training data
training.fraction <- 0.8

# Train and Test Split using randomization
sample.size <- nrow(X)
index <- sample(1:sample.size)
n.train <- floor(sample.size*training.fraction)
training.data <- X[index[1:n.train],]
testing.data <- X[-index[1:n.train],]
dim(training.data)
dim(testing.data)

# check to see if our outcome is balanced between training and testing sets
prop.table(table(training.data$EXPIRE_FLAG))
prop.table(table(testing.data$EXPIRE_FLAG))

#Predict Mortality Using Logistic Regression

Mortality = glm(EXPIRE_FLAG ~ GENDER+Ischaemic,data=training.data, family=binomial)
summary(Mortality)

#odds ratios ## odds ratios shouldn't include 1 in them!
exp(cbind(OR = round(coef(Mortality),3), round(confint(Mortality),3)))
# predict from testing
# Testing the Model
probs <- predict(Mortality, newdata = testing.data, type = "response")
pred <- ifelse(probs > 0.5, 1, 0) # chose a threshold of 0.5
pred<-as.factor(pred)
#print("Confusion Matrix for logistic regression"); table(Predicted = pred, Actual = testing.data$EXPIRE_FLAG)
confusionMatrix(pred, testing.data$EXPIRE_FLAG, positive="1" ) # Confusion Matrix for logistic regression

# ROC curve
g <- roc(EXPIRE_FLAG ~ probs, data = testing.data)
plot(g) 
