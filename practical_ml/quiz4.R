library(AppliedPredictiveModeling)
library(caret)
library(ElemStatLearn)
library(pgmm)
library(rpart)
library(gbm)
library(lubridate)
library(forecast)
library(e1071)

# Question 1
# Load the vowel.train and vowel.test data sets

data(vowel.train)
data(vowel.test) 

# Set the variable y to be a factor variable in both 
# the training and test set. Then set the seed to 33833. 
vowel.train$y <- factor(vowel.train$y)
vowel.test$y <- factor(vowel.test$y)
set.seed(33833)

# Fit (1) a random forest predictor relating the factor 
# variable y to the remaining variables and (2) a 
# boosted predictor using the "gbm" method. Fit these 
# both with the train() command in the caret package. 
mod.rf <- train(y~ .,data=vowel.train,method="rf",prox=TRUE)
mod.gbm <- train(y~ .,data=vowel.train,method="gbm",verbose=FALSE)

# What are the accuracies for the two approaches on the 
# test data set? 
pr <- predict(mod.rf, vowel.test)
pg <- predict(mod.gbm,vowel.test)


pv <- data.frame(pr,pg,y=vowel.test$y)

n <- dim(pv)[[1]]

ac1 <- sum(diag(table(pv$y, pv$pr)))/n
ac2 <- sum(diag(table(pv$y, pv$pg)))/n

# What is the accuracy among the test 
# set samples where the two methods agree? 
agree <- pv$pr == pv$pg
dim(pv[agree,]); dim(pv)
m <- dim(pv[agree,])[[1]]
ac3 <- sum(diag(table(pv[agree,]$y, pv[agree,]$pr)))/m

print(ac1); print(ac2); print(ac3)

### RF Accuracy = 0.6082
### GBM Accuracy = 0.5152
### Agreement Accuracy = 0.6361

# Question 2
# Load the Alzheimer's data using the following commands
library(gbm)
set.seed(3433)
library(AppliedPredictiveModeling)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]

# Set the seed to 62433 and predict diagnosis with all 
# the other variables using a random forest ("rf"), 
# boosted trees ("gbm") and linear discriminant analysis 
# ("lda") model. 
set.seed(62433)

m1 <- train(diagnosis ~ ., data=training, method="rf" ,prox=TRUE)
m2 <- train(diagnosis ~ ., data=training, method="gbm",verbose=FALSE)
m3 <- train(diagnosis ~ ., data=training, method="lda")

p1 <- predict(m1,training)
p2 <- predict(m2,training)
p3 <- predict(m3,training)

# Stack the predictions together using random forests 
# ("rf"). What is the resulting accuracy on the test set? 
# Is it better or worse than each of  the individual 
# predictions? 

stacked <- data.frame(p1,p2,p3,diagnosis=training$diagnosis)
model <- train(diagnosis~., data=stacked, method="rf",prox=TRUE)

t1 <- predict(m1,testing)
t2 <- predict(m2,testing)
t3 <- predict(m3,testing)
tstacked <- data.frame(p1=t1,p2=t2,p3=t3,diagnosis=testing$diagnosis)
td <- predict(model,tstacked)

n <- length(testing$diagnosis)
ac1 <- sum(diag(table(tstacked$diagnosis, tstacked$p1)))/n
ac2 <- sum(diag(table(tstacked$diagnosis, tstacked$p2)))/n
ac3 <- sum(diag(table(tstacked$diagnosis, tstacked$p3)))/n
act <- sum(diag(table(tstacked$diagnosis, td)))/n

paste("rf:",round(ac1,digits = 3),"gbm:",round(ac2,digits = 3))
paste("lda:",round(ac3,digits = 3),"stacked:",round(act,digits = 3))

### Stacked Accuracy: 0.80 is better than random forests and lda and the same as boosting. 

# Question 3
# Load the concrete data with the commands:
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]

# Set the seed to 233 and fit a lasso model to predict 
# Compressive Strength. Which variable is the last coefficient 
# to be set to zero as the penalty increases? (Hint: it may
# be useful to look up ?plot.enet). 

set.seed(233)

x <- as.matrix(subset(training,select=-c(CompressiveStrength)))
y <- as.matrix(subset(training,select=c(CompressiveStrength)))
head(x)
head(y)

modfit <- enet(x,y,lambda = 0)
plot(modfit, xvar = "penalty", use.color = TRUE)




# Question 4
# Load the data on the number of visitors to the instructors blog from here:
# https://d396qusza40orc.cloudfront.net/predmachlearn/gaData.csv
# Using the commands:

library(lubridate) # For year() function below

download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/gaData.csv",destfile = "gaData.csv")
dat = read.csv("gaData.csv")
training = dat[year(dat$date) < 2012,]
testing = dat[(year(dat$date)) > 2011,]
tstrain = ts(training$visitsTumblr)

tsvisits = ts(dat$visitsTumblr)
tstest = ts(testing$visitsTumblr)


# Fit a model using the bats() function in the forecast package 
# to the training time series. Then forecast this model for the 
# remaining time points. For how many of the testing points is 
# the true value within the 95% prediction interval bounds? 
fit <- bats(tstrain)

n <- length(tstest)

pred <- forecast(fit,h=n)
plot(pred); lines(tsvisits,col=alpha("red",alpha = 0.5)); 

inside <- 100*sum((tstest < pred$upper[,2]) & (tstest > pred$lower[,2]))/n
inside

### 96.17021

# Question 5
# Load the concrete data with the commands:
  
set.seed(3523)
library(AppliedPredictiveModeling)
data(concrete)
inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
training = concrete[ inTrain,]
testing = concrete[-inTrain,]

# Set the seed to 325 and fit a support vector machine using the 
# e1071 package to predict Compressive Strength using the default 
# settings. Predict on the testing set. What is the RMSE? 

set.seed(325)
library(e1071)
svm.model <- svm(CompressiveStrength ~ ., data = training)
p <- predict(svm.model,testing)
RMSE(p,testing$CompressiveStrength)
### 6.72