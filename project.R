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

set.seed(123)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
getwd()
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
                PolicyCount_2016,
                PolicyCount_2017,
                PolicyCount_2018,
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
                quo3 = QuoteCount_2018,
                pol1 = PolicyCount_2016,
                pol2 = PolicyCount_2017,
                pol3 = PolicyCount_2018)

prediction_data <- broker_data %>%
  dplyr::select(Submissions_2016, 
                Submissions_2017,
                Submissions_2018,
                Submissions_2019,
                QuoteCount_2016,
                QuoteCount_2017,
                QuoteCount_2018,
                QuoteCount_2019,
                PolicyCount_2016,
                PolicyCount_2017,
                PolicyCount_2018,
                PolicyCount_2019,
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
                quo4 = QuoteCount_2019,
                pol1 = PolicyCount_2016,
                pol2 = PolicyCount_2017,
                pol3 = PolicyCount_2018,
                pol4 = PolicyCount_2019)

head(model_data)

model_data[is.na(model_data)] <- 0
prediction_data[is.na(prediction_data)] <- 0

sum(is.infinite(model_data$qr1))
sum(is.infinite(model_data$qr2))
sum(is.infinite(model_data$qr3))

sum(is.infinite(prediction_data$qr1))
sum(is.infinite(prediction_data$qr2))
sum(is.infinite(prediction_data$qr3))


#### If all GWP values are missing; zero all GWP values

#*******
#######check if this makes any difference
model_all_gwp_missing <- apply(is.na(model_data),1,sum) == ncol(model_data)-1
model_data[model_all_gwp_missing,-grep("up_no", colnames(model_data))] <- 0 

#model_data <- broker_data %>%
 #dplyr::select(GWP_2016, GWP_2017, GWP_2018, GWP_2019) %>%
 #dplyr::mutate(up_no = factor(
 #if_else(GWP_2019 > GWP_2018,  "up",  "no",  missing="no"))) %>%
 #dplyr::select(-GWP_2019) %>%
 #dplyr::rename(GWP1 = GWP_2016, 
 #GWP2 = GWP_2017, 
 #GWP3 = GWP_2018) 
head(prediction_data)

#broker_id = as.numeric(broker_data$broker_name)

#Quote Ratio

prediction_data <- prediction_data %>%
  dplyr::mutate(qr1 = quo1/sub1,
                qr2 = quo2/sub2,
                qr3 = quo3/sub3,
                qr4 = quo4/sub4,
                qr_4year = (quo1+quo2+quo3+quo4)/(sub1+sub2+sub3+sub4))

model_data <- model_data %>%
  dplyr::mutate(qr1 = quo1/sub1,
                qr2 = quo2/sub2,
                qr3 = quo3/sub3,
                qr_3year = (quo1+quo2+quo3)/(sub1+sub2+sub3))

model_data$qr1[is.na(model_data$qr1)] <- 0.0
model_data$qr2[is.na(model_data$qr2)] <- 0.0
model_data$qr3[is.na(model_data$qr3)] <- 0.0
model_data$qr_3year[is.na(model_data$qr_3year)] <- 0.0

model_data$qr1[is.infinite(model_data$qr1)] <- 1.0
model_data$qr2[is.infinite(model_data$qr2)] <- 1.0
model_data$qr3[is.infinite(model_data$qr3)] <- 1.0
model_data$qr_3year[is.infinite(model_data$qr_3year)] <- 1.0

model_data[is.na(model_data)] <- 0.0

prediction_data$qr1[is.na(prediction_data$qr1)] <- 0.0
prediction_data$qr2[is.na(prediction_data$qr2)] <- 0.0
prediction_data$qr3[is.na(prediction_data$qr3)] <- 0.0
prediction_data$qr4[is.na(prediction_data$qr4)] <- 0.0
prediction_data$qr_4year[is.na(prediction_data$qr_4year)] <- 0.0

prediction_data$qr1[is.infinite(prediction_data$qr1)] <- 1.0
prediction_data$qr2[is.infinite(prediction_data$qr2)] <- 1.0
prediction_data$qr3[is.infinite(prediction_data$qr3)] <- 1.0
prediction_data$qr4[is.infinite(prediction_data$qr4)] <- 1.0
prediction_data$qr_4year[is.infinite(prediction_data$qr_4year)] <- 1.0

prediction_data[is.na(prediction_data)] <- 0.0

#Add Hit Ratio = policycount/quotecount

prediction_data <- prediction_data %>%
  dplyr::mutate(hr1 = pol1/quo1,
                hr2 = pol2/quo2,
                hr3 = pol3/quo3,
                hr4 = pol4/quo4,
                hr_4year = (pol1+pol2+pol3+pol4)/(quo1+quo2+quo3+quo4))

model_data <- model_data %>%
  dplyr::mutate(hr1 = pol1/quo1,
                hr2 = pol2/quo2,
                hr3 = pol3/quo3,
                hr_3year = (pol1+pol2+pol3)/(quo1+quo2+quo3))

model_data[is.na(model_data)] <- 0.0
prediction_data[is.na(prediction_data)] <- 0.0

model_data$hr1[is.infinite(model_data$hr1)] <- 1.0
model_data$hr2[is.infinite(model_data$hr2)] <- 1.0
model_data$hr3[is.infinite(model_data$hr3)] <- 1.0
model_data$hr_3year[is.infinite(model_data$hr_3year)] <- 1.0

prediction_data$hr1[is.infinite(prediction_data$hr1)] <- 1.0
prediction_data$hr2[is.infinite(prediction_data$hr2)] <- 1.0
prediction_data$hr3[is.infinite(prediction_data$hr3)] <- 1.0
prediction_data$hr4[is.infinite(prediction_data$hr4)] <- 1.0
prediction_data$hr_4year[is.infinite(prediction_data$hr_4year)] <- 1.0

#Add Success Ratio = policycount/submissions

prediction_data <- prediction_data %>%
  dplyr::mutate(sr1 = pol1/sub1,
                sr2 = pol2/sub2,
                sr3 = pol3/sub3,
                sr4 = pol4/sub4,
                sr_4year = (pol1+pol2+pol3+pol4)/(sub1+sub2+sub3+sub4))

model_data <- model_data %>%
  dplyr::mutate(sr1 = pol1/sub1,
                sr2 = pol2/sub2,
                sr3 = pol3/sub3,
                sr_3year = (pol1+pol2+pol3)/(sub1+sub2+sub3))

model_data[is.na(model_data)] <- 0.0

model_data$sr1[is.infinite(model_data$sr1)] <- 1.0
model_data$sr2[is.infinite(model_data$sr2)] <- 1.0
model_data$sr3[is.infinite(model_data$sr3)] <- 1.0
model_data$sr_3year[is.infinite(model_data$sr_3year)] <- 1.0

prediction_data$sr1[is.infinite(prediction_data$sr1)] <- 1.0
prediction_data$sr2[is.infinite(prediction_data$sr2)] <- 1.0
prediction_data$sr3[is.infinite(prediction_data$sr3)] <- 1.0
prediction_data$sr4[is.infinite(prediction_data$sr4)] <- 1.0
prediction_data$sr_4year[is.infinite(prediction_data$sr_4year)] <- 1.0

prediction_data[is.na(prediction_data)] <- 0.0

#rpart model

#model_data <- broker_data %>%
# dplyr::select(GWP_2016, GWP_2017, GWP_2018, GWP_2019) %>%
#dplyr::mutate(up_no = factor(
# if_else(GWP_2019 > GWP_2018,  "up",  "no",  missing="no"))) %>%
# dplyr::select(-GWP_2019) %>%
# dplyr::rename(GWP1 = GWP_2016, 
#GWP2 = GWP_2017, 
# GWP3 = GWP_2018) 

#summary(as.factor(model_data$up_no))

#model_all_gwp_missing <- apply(is.na(model_data),1,sum) == ncol(model_data)-1
#model_data[model_all_gwp_missing,-grep("up_no", colnames(model_data))] <- 0 



#prediction_data <- broker_data %>%
#dplyr::select(GWP_2017, GWP_2018, GWP_2019) %>%
#dplyr::rename(GWP1 = GWP_2017, 
# GWP2 = GWP_2018, 
#GWP3 = GWP_2019) 

prediction_all_gwp_missing <- apply(is.na(prediction_data),1,sum) == ncol(prediction_data)
prediction_data[prediction_all_gwp_missing,] <- 0 


train_rows <- createDataPartition(model_data$up_no,
                                  p=0.75,
                                  list=FALSE)
train_broker <- model_data[train_rows,]
test_broker <- model_data[-train_rows,]

#rpart model

rpart_broker <- rpart(up_no ~ ., data=train_broker)
rpart_broker_predict <- predict(rpart_broker, test_broker, type="prob")
rpart_broker_prediction <- prediction(rpart_broker_predict[,2], 
                                      test_broker$up_no,
                                      label.ordering=c("no", "up"))
rpart_broker_performance <- performance(rpart_broker_prediction, "tpr", "fpr")
rpart_broker_auc <- performance(rpart_broker_prediction, "auc")
rpart_broker_auc@y.values[[1]]

#write.csv(rpart_broker_2020, 
#file="predictions2.csv",
#quote=FALSE, 
#row.names=FALSE)

plot(rpart_broker_performance, col=1)

#NN Model

modelLookup("nnet")
my_weights <- rep(2, nrow(train_broker))
my_weights[train_broker$up_no == "up"] <- 2
my_weights[train_broker$up_no == "down"] <- 4

broker_NN <- train(up_no ~ .,
                   data=train_broker,
                   method="nnet",
                   metric="ROC",
                   weights=my_weights,
                   trControl=trainControl(classProbs = TRUE, summaryFunction = twoClassSummary))

broker_NN

my_nn_predict <- predict(broker_NN, newdata=test_broker, type="prob")
my_nn_pred <- prediction(my_nn_predict[,1], 
                         test_broker$up_no,
                         label.ordering=c("up", "no"))
my_nn_perf <- performance(my_nn_pred, "tpr", "fpr")

nn_broker_auc <- performance(my_nn_pred, "auc")
nn_broker_auc@y.values[[1]]

plot(my_nn_perf, col=4, add=TRUE)


##RF

broker_rf <- randomForest(up_no ~ ., 
                          data = train_broker, 
                          classwt=c(2,1),
                          importance=TRUE)

broker_rf$importance

broker_predict_rf <- predict(broker_rf, newdata=test_broker, type="class")
(broker_rf_confusion <- table(test_broker$up_no, broker_predict_rf))

my_rf_predict <- predict(broker_rf, newdata=test_broker, type="prob")
my_rf_pred <- prediction(my_rf_predict[,1], 
                         test_broker$up_no,
                         label.ordering=c("up", "no"))
my_rf_perf <- performance(my_rf_pred, "tpr", "fpr")

plot(my_rf_perf, col=2, add=TRUE)

rf_broker_auc <- performance(my_rf_pred, "auc")
rf_broker_auc@y.values[[1]]

##Predicted Model (RF)

broker_rf_prediction <- randomForest(up_no ~ ., 
                          data = prediction_data, 
                          classwt=c(2,1),
                          importance=TRUE)
broker_rf$importance

my_rf_predict <- predict(broker_rf_prediction, newdata=prediction_data, type="prob")
my_rf_pred <- prediction(my_rf_predict[,1], prediction_data$up_no, label.ordering = c("up","no"))
my_rf_perf <- performance(my_rf_pred, "tpr", "fpr")

rf_broker_auc <- performance(my_rf_pred, "auc")
rf_broker_auc@y.values[[1]]


my_rf_2020 <- predict(broker_rf, prediction_data, type="class")

output_predictions_2020 <- data.frame(broker_id = rownames(prediction_data), prediction = my_rf_2020[,2])

write.csv(output_predictions_2020, file = "predictions_rf.csv", quote = FALSE, row.names = FALSE)
