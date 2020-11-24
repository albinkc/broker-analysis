# Apply a clustering method at least three years of information ending with 2018 to cluster brokers into five segments. Describe the characteristic properties of each segment. Provide a visualization of your broker segmentation using principal component analysis and describe the clusters in terms of the components.

library(dplyr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

broker_data = read.table("alchemy_broker_data.csv", 
                          header=TRUE,
                          colClasses = c(rep("character",3), rep("numeric", 29)),
                          sep=",",
                          row.names=1)

broker_data <- broker_data %>%
  dplyr::select(Submissions_2016,
                Submissions_2017,
                Submissions_2018,
                QuoteCount_2016,
                QuoteCount_2017,
                QuoteCount_2018,
                PolicyCount_2016,
                PolicyCount_2017,
                PolicyCount_2018,
                GWP_2016,
                GWP_2017, 
                GWP_2018,
                AvgTIV_2016,
                AvgTIV_2017,
                AvgTIV_2018)

#Handling NA's

#Remove rows with more than 10 NA's

broker_data = broker_data[rowSums(is.na(broker_data)) <= 10,]

summary(broker_data)
##Impute NA's with 0
broker_data[is.na(broker_data)] = 0


#Remove outliers

library('outliers')
outliers <- apply(broker_data[colnames(broker_data)],2,function(x) which(x == outlier(x)))
broker_data <- broker_data[-unique(unlist(outliers)),]

#Scaling

broker_data_scaled = scale(broker_data)

#Clustering

broker_kmeans = kmeans(broker_data_scaled, centers=5)
summary(broker_kmeans)

#PCA

broker_data_pca <- prcomp(broker_data_scaled, retx=TRUE)
broker_data_pca$rotation[,1:2]
plot(broker_data_pca$x[,1:2], col=broker_kmeans$cluster, pch=broker_kmeans$cluster)

