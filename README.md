# ChurnPrediction_GLM_RF_EnsembleGLM-RF
We are going to predict churn for a telecom operator. We will learn resampling method using caret and generate your first ensemble model made up with a GLM and a Random Forest
setwd("C:/Users/Install/Desktop/CASE_R_september")

#summarize the target variable
library(caret)
library(C50)
data(churn)
table(churnTrain$churn) / nrow(churnTrain) 

churn_x <- churnTrain$churn
churn_y <- churnTest$churn

########################################################
#Make custom train/test indices
########################################################

#STEP1 first order of business is to create a reusable trainControl object#
#you can use to reliably compare them.

# Create custom indices: myFolds
myFolds <- createFolds(churn_y, k = 5)

# Create reusable trainControl object: myControl
myControl <- trainControl(
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE,
  savePredictions = TRUE,
  index = myFolds
)
########################################################
#Fit the baseline model
########################################################
#glmnet, which penalizes linear and logistic regression models 
#on the size and number of coefficients to help prevent overfitting.
# Fit glmnet model: model_glmnet
model_glmnet <- train(
  churn ~., data = churnTrain,
  metric = "ROC",
  method = "glmnet",
  tuneGrid = expand.grid(
    alpha = 0:1,
    lambda = 0:10/10
  ),
  trControl = myControl
)
#plot the results
plot(model_glmnet)
#plot the coefficients
plot(model_glmnet$finalModel)

########################################################
#Random forest with custom trainControl
########################################################
#What RF's drawback ?
#You no longer have model coefficients to help interpret the model.
#RFcombines an ensemble of non-linear decision trees into a highly flexible
library(ranger)
#ranger package, which is a re-implementation of randomForest 
#that produces almost the exact same results, but is faster

set.seed(42)
churnTrain$churn <- factor(churnTrain$churn, levels = c("no", "yes"))
model_rf <- train(
  churn ~., churnTrain,
  metric = "ROC",
  method = "ranger",
  trControl = myControl
)
#plot RF
plot(model_rf)


#What's the primary reason that train/test indices need to match when comparing two models?
#Because otherwise you wouldn't be doing a fair comparison of your models and your results could be due to chance.

########################################################
#Create a resamples object
########################################################
#Now that you have fit two models to the churn dataset,
#it's time to compare their out-of-sample predictions
#and choose which one is the best model for your dataset.

# Create model_list
model_list <- list(item1 = model_glmnet, item2 = model_rf)
# Pass model_list to resamples(): resamples
resamples <- resamples(model_list)
# Summarize the results
summary(resamples)


########################################################
#Create a box-and-whisker plot
########################################################
#box-and-whisker plot, which allows you to compare the distribution of predictive accuracy 
#(in this case AUC) for the two models.

#you want the model with the higher median AUC, as well as a smaller range between min and max AUC.
# Create bwplot
bwplot(resamples, metric = "ROC")
#RF has an higher median AUC and a smaller ranger between min & max AUC
########################################################
#Create a scatter-plot
########################################################
#This plot shows you how similar the two models' performances are on different folds.
#useful for identifying if one model is consistently better than the other across all folds
#or if there are situations when the inferior model produces better predictions on a particular subset of the data
# Create xyplot
xyplot(resamples, metric = "ROC")

########################################################
#Ensembling models
########################################################
library(caretEnsemble)
#caretEnsemble provides the caretList() function 
#for creating multiple caret models at once on the same dataset,
#create a CaretList
model_list <- caretList(
  churn ~. , 
  data = churnTrain,
  methodList = c("glm","rf"),
  trControl = myControl)

# Create ensemble model: stack
stack <- caretStack(model_list, method = "glm")

# Look at summary
summary(stack)
