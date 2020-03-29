## Predicting zinc/bromine electrolyte properties and complex proportions using Neural Network modelling ----
# Overview: Zinc/bromine flow battery electrolyte contains various proportions of complexes as a function of pH and conductivity.
# Overview (continued): Prdicting some electrolyte properties can be challenging at high salt concentrations.
# Overview (continued): It is helpful to be able to predict properties and proportions as they may have implications for battery performance.

# This code: Uses ZnBr2 and ZnCl2 salt concentrations to predict pH, as an example
# Dataset: Zinc/Bromine Flow Battery Electrolyte Properties and Complex Proportions
# Dataset source: Dr Gobinath Pillai Rajarathnam, "The Zinc/bromine Flow Battery: Fundamentals and Novel Materials for Technology Advancement", 2016, PhD Thesis, The University of Sydney.

# Author: Dr Gobinath Pillai Rajarathnam
# Google Scholar: https://scholar.google.com.au/citations?user=7mbZHrcAAAAJ&hl=en
# GitHub: https://github.com/machine-energy
# LinkedIn: https://au.linkedin.com/in/gobinath-rajarathnam-0364a910b

## initialisation ----
# load libraries
library(boot)
library(data.table)
library(dataPreparation) # for scaling data
library(dplyr)
library(neuralnet) # for neural network modelling
library(plyr)
library(readr) # to import csv files

# initialisation
setwd("~/Downloads") # Set working directory
set.seed(10) # set the seed (in this case 10) for reproducibility of results

## dataset preparation ----
# load the dataset
data.full <- read_csv("Raman Dataset - GPR rev01.csv") # import dataset
data <- data.full[,c(2,3,8)] # subset coluns with ZnBr2 and ZnCl2 salt concentrations, and pH values

# split Train/Validation and Test data
subset.size <- floor(nrow(data)*0.75) # in this case, makes note of value for 75% of the dataset's number of rows
training.data.rows <- sample(seq_len(nrow(data)), size=subset.size) # makes note of which rows to use for Training subset of dataset
training.data <- data[training.data.rows, ] # create Training subset from full dataset
test.data <- data[-training.data.rows, ] # create Test subset from full dataset

# calculate and apply scaling
data.scales <- build_scales(dataSet = training.data) # determine scale from Training dataset
train.scaled <- fastScale(dataSet = training.data, scales = data.scales, way = "scale") # apply Training dataset scale to Training dataset
test.scaled <- fastScale(dataSet = test.data, scales = data.scales, way = "scale") # apply Training dataset scale to Test dataset

## preparing and training the neural network ----
# setting global variables
nodes1stlayer <- 6 # number of nodes in 1st layer (closer to input)
nodes2ndlayer <- 4 # number of nodes in 2nd layer (closer to output)
kfold <- 10 # many-fold cross-validation to increase robustness of model (in this case 10 times)
holdback <- 0.6 # how much of the data to use in training set (in this case, 60%)

# setup of neural network formula
f <- as.formula(paste0("pH ~ ZnBr2saltM + ZnCl2saltM"))

# create and start a progress bar to monitor model development
pbar <- create_progress_bar('text')
pbar$init(kfold)

# neural network training loop
for(i in 1:kfold){
  index <- sample(1:nrow(train.scaled),round(holdback*nrow(train.scaled))) # indexing the data
  train.cv <- train.scaled[index,] # create a training set using the indexed sample
  nn <- neuralnet(f, data = train.cv, hidden = c(nodes1stlayer,nodes2ndlayer), linear.output = FALSE, act.fct = "tanh") # training the neural network model
  pbar$step() # updates the progress bar
}

# plot neural network
plot(nn)

## model results extraction and post-processing ----
pr.nn <- compute(nn,train.scaled[,1:2]) # predicting across the Training dataset
pr.nn <- as.data.frame(pr.nn$net.result) # extract predicted result
colnames(pr.nn) <- "pH"
pr.nn <- fastScale(dataSet = pr.nn, scales = data.scales[3], way = "unscale") # unscale dataset
colnames(pr.nn) <- "Predicted pH (Training)" # rename predicted pH column
merged.results.train <- bind_cols(training.data, pr.nn) # combine predicted and actual pH in Training dataset
cv.error.train <- sum((training.data[3] - training.data[4])^2)/nrow(training.data) # calculate cross-validation error
cv.error.train # display cross-validation error
write.csv(merged.results.train, "Electrolyte NN Results - Training.csv") # write results to a csv file for collaboration/sharing/records

pr.nn <- compute(nn,test.scaled[,1:2]) # predicting across the Test dataset
pr.nn <- as.data.frame(pr.nn$net.result) # extract predicted result
colnames(pr.nn) <- "pH"
pr.nn <- fastScale(dataSet = pr.nn, scales = data.scales[3], way = "unscale") # unscale dataset
colnames(pr.nn) <- "Predicted pH (Test)" # rename predicted pH column
merged.results.test <- bind_cols(test.data, pr.nn) # combine predicted and actual pH in Test dataset
cv.error.test <- sum((test.data[3] - test.data[4])^2)/nrow(test.data) # calculate cross-validation error
cv.error.test # display cross-validation error
write.csv(merged.results.train, "Electrolyte NN Results - Test.csv") # write results to a csv file for collaboration/sharing/records