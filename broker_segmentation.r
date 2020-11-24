# Apply a clustering method at least three years of information ending with 2018 to cluster brokers into five segments. Describe the characteristic properties of each segment. Provide a visualization of your broker segmentation using principal component analysis and describe the clusters in terms of the components.

library(dplyr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

broker_data = read.table("alchemy_broker_data.csv", 
                          header=TRUE,
                          colClasses = c(rep("character",3), rep("numeric", 29)),
                          sep=",",
                          row.names=1)

broker_data <- broker_data %>%
  dplyr::select(Submissions_2017,
                Submissions_2018,
                Submissions_2019,
                QuoteCount_2017,
                QuoteCount_2018,
                QuoteCount_2019,
                PolicyCount_2017,
                PolicyCount_2018,
                PolicyCount_2019,
                GWP_2017,
                GWP_2018, 
                GWP_2019,
                AvgTIV_2017,
                AvgTIV_2018,
                AvgTIV_2019)

#Handling NA's

#Remove rows with more than 10 NA's

broker_data = broker_data[rowSums(is.na(broker_data)) <= 10,]

colnames(broker_data)


#Remove outliers

#Clustering

broker_kmeans = kmeans(lending_data_sub, centers=5)

