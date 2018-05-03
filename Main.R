library(data.table)
library(zoo)
library(forecast)
library(ggplot2)
library(xgboost)
version
install.packages('devtools')
library(devtools)
install_github('andreacirilloac/updateR')
library(updateR)


#we retrieve the data 
setwd("C:/Users/Saana/Desktop/6162_Final")
getwd()
trainingData <- read.csv("train.csv")
testingData <- read.csv("test.csv")
storeData <- read.csv("store.csv")

# we merge storeData to trainingData and testingData
trainingData <- merge(trainingData, storeData, by="Store")
testingData <- merge(testingData, storeData, by="Store")

#clean out the data
str(trainingData)
str(testingData)
#data contains some NA in some feilds which can be converted to 0
trainingData[is.na(trainingData)] <- 0
testingData[is.na(testingData)] <- 0
str(trainingData)
str(testingData)

#getting the date as Date format to help with splitting
testingData$Date <- as.Date(testingData$Date)
trainingData$Date <- as.Date(trainingData$Date)

str(trainingData)

#Split and format the dates into days, months and years 
testingData$month <- as.integer(format(testingData$Date, "%m"))
testingData$year <- as.integer(format(testingData$Date, "%y"))
testingData$day <- as.integer(format(testingData$Date, "%d"))

trainingData$month <- as.integer(format(trainingData$Date, "%m"))
trainingData$year <- as.integer(format(trainingData$Date, "%y"))
trainingData$day <- as.integer(format(trainingData$Date, "%d"))

names(trainingData)
names(testingData)

write.csv(trainingData, file = "trainingDataCleaned.csv")
write.csv(testingData, file = "testingDataCleaned.csv")

# we add feature names
feature.names <- names(trainingData)[c(1,2,6,8:12,14:19)]
feature.names

# we replace all the text variables with numeric id
for (f in feature.names)
{
  if (class(trainingData[[f]]) == "character")
  {
    levels <- unique(c(trainingData[[f]], testingData[[f]]))
    trainingData[[f]] <- as.integer(factor(trainingData[[f]], levels=levels))
    testingData[[f]]  <- as.integer(factor(testingData[[f]],  levels=levels))
  }
}

#cut the feature and assign to a variable tra
trainSet<-trainingData[,feature.names]

RMPSE<- function(preds, dtrain) {
  labels1 <- getinfo(dtrain, "label")
  labels<-labels1[labels1>0]
  elab<-exp(as.numeric(labels))-1
  epreds<-exp(as.numeric(preds[labels1>0]))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}

hght<-sample(nrow(trainingData),10000)

dval<-xgb.DMatrix(data=data.matrix(trainSet[hght,]),label=log(trainingData$Sales+1)[hght])
dtrain<-xgb.DMatrix(data=data.matrix(trainSet[-hght,]),label=log(trainingData$Sales+1)[-hght])
watchlist<-list(val=dval,trainingData=dtrain)

param <- list( objective = "reg:linear", booster = "gbtree",
                eta = 0.25, max_depth = 8, subsample = 0.7, colsample_bytree = 0.7 )

clf <- xgb.train( params = param, data = dtrain, nrounds = 700,
                    verbose = 1, early.stop.round = 30, watchlist = watchlist, maximize = FALSE, feval = RMPSE)

forecst <- exp(predict(clf, data.matrix(testingData[,feature.names]))) -1
results <- data.frame(Id = testingData$Id, Sales = forecst)

#write the reaults to a .csv file
write.csv(results, file = "results.csv")