library(rpart) #classification trees
library(caret) #partition data and missing values
library(ROCR)  #plotting ROC curve
library(dplyr) #slice and dice data
library(dummies)
library(randomForest)
library(e1071)
library(kernlab)
library(nnet)
library(NeuralNetTools)

set.seed(12345)

broker_data <- read.table("C:\\Users\\jakob\\Desktop\\SCMA 648\\alchemy_broker_data.csv", 
                          header=TRUE,
                          colClasses = c(rep("character",3), rep("numeric", 29)),
                          sep=",",
                          row.names=1)

#rpart model

model_data <- broker_data %>%
  dplyr::select(GWP_2016, GWP_2017, GWP_2018, GWP_2019) %>%
  dplyr::mutate(up_no = factor(
    if_else(GWP_2019 > GWP_2018,  "up",  "no",  missing="no"))) %>%
  dplyr::select(-GWP_2019) %>%
  dplyr::rename(GWP1 = GWP_2016, 
                GWP2 = GWP_2017, 
                GWP3 = GWP_2018) 

summary(as.factor(model_data$up_no))

model_all_gwp_missing <- apply(is.na(model_data),1,sum) == ncol(model_data)-1
model_data[model_all_gwp_missing,-grep("up_no", colnames(model_data))] <- 0 



prediction_data <- broker_data %>%
  dplyr::select(GWP_2017, GWP_2018, GWP_2019) %>%
  dplyr::rename(GWP1 = GWP_2017, 
                GWP2 = GWP_2018, 
                GWP3 = GWP_2019) 

prediction_all_gwp_missing <- apply(is.na(prediction_data),1,sum) == ncol(prediction_data)
prediction_data[prediction_all_gwp_missing,] <- 0 


train_rows <- createDataPartition(model_data$up_no,
                                  p=0.75,
                                  list=FALSE)
train_broker <- model_data[train_rows,]
test_broker <- model_data[-train_rows,]

rpart_broker <- rpart(up_no ~ ., data=train_broker)
rpart_broker_predict <- predict(rpart_broker, test_broker, type="prob")
rpart_broker_prediction <- prediction(rpart_broker_predict[,2], 
                                      test_broker$up_no,
                                      label.ordering=c("no", "up"))
rpart_broker_performance <- performance(rpart_broker_prediction, "tpr", "fpr")
rpart_broker_auc <- performance(rpart_broker_prediction, "auc")
rpart_broker_auc@y.values[[1]]


rpart_broker_2020_predict <- predict(rpart_broker, 
                                     prediction_data,
                                     type="prob")

rpart_broker_2020 <- data.frame(broker_id = rownames(prediction_data),
                                prediction = rpart_broker_2020_predict[,2])

write.csv(rpart_broker_2020, 
          file="predictions.csv",
          quote=FALSE, 
          row.names=FALSE)

plot(rpart_broker_performance, col=1)



#NN Model

modelLookup("nnet")
my_weights <- rep(2, nrow(train_broker))
my_weights[train_broker$up_no == "up"] <- 1

broker_NN <- train(up_no ~ .,
                   data=train_broker,
                   method="nnet",
                   metric="ROC",
                   weights=my_weights,
                   trControl=trainControl(classProbs = TRUE, summaryFunction = twoClassSummary))

broker_NN

broker_NN_predict <- predict(broker_NN, newdata=test_)


##Ratios and other stuff

set.seed(12345)

broker_data <- read.table("alchemy_broker_data.csv", 
                          header=TRUE,
                          colClasses = c(rep("character",3), rep("numeric", 29)),
                          sep=",",
                          row.names=1)


model_data <- broker_data %>%
  dplyr::select(Submissions_2016, 
                Submissions_2017,
                Submissions_2018,
                QuoteCount_2016,
                QuoteCount_2017,
                QuoteCount_2018,
                GWP_2018, 
                GWP_2019) %>%
  dplyr::mutate(up_no = factor(
    if_else(GWP_2019 > GWP_2018,  "up",  "no",  missing="no"))) %>%
  dplyr::select(-GWP_2019, -GWP_2018) %>%
  dplyr::rename(sub1 = Submissions_2016,
                sub2 = Submissions_2017,
                sub3 = Submissions_2018,
                quo1 = QuoteCount_2016,
                quo2 = QuoteCount_2017,
                quo3 = QuoteCount_2018)
## in base R,
# names(model_data)[names(model_data) == "Submissions_2016"] <- sub1
# names(model_data)[names(model_data) == "Submissions_2017"] <- sub2
# ...

head(model_data)

model_data[is.na(model_data)] <- 0

model_data <- model_data %>%
  dplyr::mutate(qr1 = quo1/sub1,
                qr2 = quo2/sub2,
                qr3 = quo3/sub3,
                qr_3year = (quo1+quo2+quo3)/(sub1+sub2+sub3))

## in base R,
# model_data$qr1 <- model_data$quo1/model_data$sub1
# model_data$qr2 <- model_data$quo2/model_data$sub2
# ...
# model_data$qr_3year <- (model_data$quo1+model_data$quo2+model_data$quo3)/(model_data$sub1+model_data$sub2+model_data$sub3)

head(model_data)
summary(model_data)

model_data$qr1[is.na(model_data$qr1)] <- 0.0

model_data$qr1[is.infinite(model_data$qr1)] <- 1.0



## 1. generate a data frame called prediction_data with data through 2019 and calculate the quote ratio that can be used in a model for 2020 predictions

prediction_data <- broker_data %>%
  dplyr::select(Submissions_2016, 
                Submissions_2017,
                Submissions_2018,
                Submissions_2019,
                QuoteCount_2016,
                QuoteCount_2017,
                QuoteCount_2018,
                QuoteCount_2019,
                GWP_2018, 
                GWP_2019) %>%
  dplyr::mutate(up_no = factor(
    if_else(GWP_2019 > GWP_2018,  "up",  "no",  missing="no"))) %>%
  dplyr::select(-GWP_2019, -GWP_2018) %>%
  dplyr::rename(sub1 = Submissions_2016,
                sub2 = Submissions_2017,
                sub3 = Submissions_2018,
                sub4 = Submissions_2019,
                quo1 = QuoteCount_2016,
                quo2 = QuoteCount_2017,
                quo3 = QuoteCount_2018,
                quo4 = QuoteCount_2019)

head(prediction_data)

prediction_data <- prediction_data %>%
  dplyr::mutate(qr1 = quo1/sub1,
                qr2 = quo2/sub2,
                qr3 = quo3/sub3,
                qr4 = quo4/sub4,
                qr_4year = (quo1+quo2+quo3+quo4)/(sub1+sub2+sub3+sub4))