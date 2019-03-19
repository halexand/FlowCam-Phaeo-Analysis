# Set working directory
setwd('./region-props/')
set.seed(13)

# Load libraries
library(dplyr)
library(randomForest)
library(e1071)
library(gbm)
library(xgboost)

## Summary:
# PCA + logistic: 85-89%
# SVM: 91%
# Random forest: 94%
# GBM: 94%
# XGB: 94%

###############################################################################
# Read and organise data
other_train <- read.csv('Other_training.csv')
phaeo_train <- read.csv('Phaeo_training.csv')
other_test <- read.csv('Other_test.csv')
phaeo_test <- read.csv('Phaeo_test.csv')

other_train$type <- 'other'; other_train$type <- as.factor(other_train$type)
phaeo_train$type <- 'phaeo'; phaeo_train$type <- as.factor(phaeo_train$type)
other_test$type <- 'other'; other_test$type <- as.factor(other_test$type)
phaeo_test$type <- 'phaeo'; phaeo_test$type <- as.factor(phaeo_test$type)

dat_train <- rbind(other_train, phaeo_train)
dat_train$type <- as.numeric(dat_train$type) - 1
# dat_train$aspect_ratio <- dat_train$major_axis_length / dat_train$minor_axis_length
str(dat_train)
summary(dat_train)

# NAs present in sub-area parameters. Remove as cleaning step
dat_train <- filter(dat_train, subregion_mean_area != 'NaN') 
dat_train <- filter(dat_train, blue_color_std != 'NaN') 
summary(dat_train)


dat_test <- rbind(other_test, phaeo_test)
dat_test$type <- as.numeric(dat_test$type) - 1
dat_test$type <- as.factor(dat_test$type)
# dat_test$aspect_ratio <- dat_test$major_axis_length / dat_test$minor_axis_length
str(dat_test)
summary(dat_test)

# NAs present in sub-area parameters. Remove as cleaning step
dat_test <- filter(dat_test, subregion_mean_area != 'NaN') 
dat_test <- filter(dat_test, blue_color_std != 'NaN') 
summary(dat_test)

###############################################################################
### Random forests

# Fit model
fc_rf <- randomForest(as.factor(type) ~ ., data = dat_train[,-c(1)], 
                      importance = TRUE, ntree = 1001)

summary(fc_rf)
plot(fc_rf)
importance(fc_rf)

# Evaluate with test set
rf_test <- predict(fc_rf, dat_test)
rf_test_results <- data.frame(round(as.numeric(rf_test)) - 1, dat_test$type)

# Confusion matrix 
(table(rf_test_results))/dim(dat_test)[1]

### GBM

# Fit model
fc_gbm <- gbm(formula = type ~ .,
              distribution = 'bernoulli',
              data = dat_train[,-c(1)],
              n.trees = 10000,
              interaction.depth = 5,
              shrinkage = 0.001,
              bag.fraction = 0.66,
              cv.folds = 10,
              keep.data = TRUE)

summary(fc_gbm)
fc_gbm
plot(fc_gbm)

# Plot for project 
# library(cowplot)
# pdat <- summary(fc_gbm)
# pdat2 <- droplevels(pdat[1:10,])
# p <- ggplot(data=pdat, aes(x=reorder(var, rel.inf), y=rel.inf)) + 
#   geom_bar(stat="identity") + ylab('Relative influence') + xlab('') +
#   coord_flip() + theme(axis.text = element_text(size = 13))
# p

# Evaluate with test set
gbm_test <- predict(object = fc_gbm, newdata = dat_test,
                                   n.trees = 10000,
                                   type = "response")
gbm_test_results <- data.frame(round(gbm_test), dat_test$type)

# Confusion matrix 
(table(gbm_test_results))/dim(dat_test)[1]

### SVM

# Fit model
fc_svm <- svm(as.factor(type) ~ ., data = dat_train[,-c(1)], 
                 method = "C-classification", kernel = "polynomial")
fc_svm

# Evaluate with test set
pred_svm <- predict(fc_svm, dat_test)
mean(pred_test == dat_test$type)
# 0.908

svm_test_results <- data.frame(pred_svm, dat_test$type)

# Confusion matrix 
(table(svm_test_results))/dim(dat_test)[1]

### XGBoost
# Prepare dataset for package requirements
dtrain <- xgb.DMatrix(data = as.matrix(dat_train[,-c(1,34)]),label = dat_train$type) 
dtest <- xgb.DMatrix(data = as.matrix(dat_test[,-c(1,34)]), label = dat_test$type) 

# Parameters for model
params <- list(booster = "gbtree", objective = "binary:logistic", eta = 0.1, 
               gamma = 0, max_depth = 6, min_child_weight = 1, subsample = 0.5, 
               colsample_bytree = 0.5)

# Cross validate model to ID best tree
xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 500, nfold = 5, 
                 showsd = T, stratified = T, print_every_n = 10, 
                 early_stopping_rounds = 50, maximize = F)

# Train best model
xgb1 <- xgb.train (params = params, data = dtrain, nrounds = xgbcv$best_iteration, 
                   watchlist = list(val=dtest,train=dtrain), print_every_n = 10, 
                   early_stopping_rounds = 10, maximize = F , eval_metric = "error")

xgbpred <- predict(xgb1,dtest)
xgb_test_results <- data.frame(round(xgbpred), dat_test$type)

(table(xgb_test_results))/dim(dat_test)[1]


###############################################################################
phaeofiles <- list.files(path = '/Volumes/Ebisu/2018-NSF-Antarctica/Project/ImageProcessing/data/processed-data/region_props_csv/', full.names = TRUE)

for (i in 1:length(phaeofiles)){
  print(i)
  dat <- read.csv(phaeofiles[i])
  preds <- round(predict(object = fc_gbm, newdata = dat, n.trees = 10000,
                type = "response"))
  preds[preds == 1] <- 'Phaeocystis'
  preds[preds == 0] <- 'Other'
  newdat <- data.frame(dat, preds)

  write.csv(newdat, paste0(phaeofiles[i]), row.names = FALSE)

}



