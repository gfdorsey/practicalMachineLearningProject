library(caret)
library(rattle)

train <- read.csv("pml-training.csv", na.strings = c('', NA))

# remove 'X' column, user_name, timestamps, and window columns
train <- train[,-c(1:7)]

# remove NA columns (153 variables -> 53 variables)
train <- train[, colSums(is.na(train)) == 0]

summary(train)
#train$cvtd_timestamp <- strptime(train$cvtd_timestamp, '%d/%m/%Y %H:%M')

test <- read.csv("pml-testing.csv", na.strings = c('', NA))

# remove 'X' column, user_name, timestamps, window, and NA columns
test <- test[,-c(1:7)]
test <- test[, colSums(is.na(test)) == 0]
summary(test)

# start with simple rpart model
fit.rpart.cv <- train(classe ~ ., data = train, method = 'rpart',
                   trControl = trainControl(method = 'cv'))
print(fit.rpart.cv)

fancyRpartPlot(fit.rpart.cv$finalModel)
# accuracy is poor (0.5) and unless the tree is truncated,
# it never predicts class D

# probably not worth the trouble, but to fill out the table, try with pca
fit.rpart.pca.cv <- train(classe ~ ., data = train, method = 'rpart',
                          preProcess = c('pca'),
                          trControl = trainControl(method = 'cv'))
print(fit.rpart.pca.cv)
# accuracy  dropped to 0.398

# try bagging
# fit.bag.cv <- train(classe ~ ., data = train, method = 'bag',
#                  trControl = trainControl(method = 'cv'))
# errors and warnings--accuracy & kappa are all NA & NaN
# need to investigate more, but skip for now

# try combination model
# fit.gam.pca.cv <- train(classe ~ ., data = train, method = 'gam',
#                         preProcess = c('pca'),
#                         trControl = trainControl(method = 'cv'))
# stopped after 24 hours
# in retrospect, this probably needed to have multiple models manually combined
# first--not sure how it would work on the raw data

# try clustering with k-nearest neighbors using pca for preprocessing
fit.knn.pca.cv <- train(classe ~ ., data = train, method = 'knn',
                    preProcess = c('pca'),
                    trControl = trainControl(method = 'cv'))
print(fit.knn.pca.cv)
# accuracy is respectable--0.971 with k=5

# try random forests to see if it improves on CART
fit.rf.pca.cv <- train(classe ~ ., data = train, method = 'rf',
                       preProcess = c('pca'),
                       trControl = trainControl(method = 'cv'))
print(fit.rf.pca.cv)
## accuracy is 0.983 with mtry = 2

## also redo rf without pca
fit.rf.cv <- train(classe ~ ., data = train, method = 'rf',
                       trControl = trainControl(method = 'cv'))
print(fit.rf.cv)
## accuracy was 0.995 with mtry = 27 (only slightly less with mtry = 2)


# try naive bayesian
fit.nb.cv <- train(classe ~ ., data = train, method = 'nb',
                   trControl = trainControl(method = 'cv'))

print(fit.nb.cv)
## accuracy is 0.744 with fL = 0 and usekernel = TRUE
## perhaps try preprocessing with pca

fit.nb.pca.cv <- train(classe ~ ., data = train, method = 'nb',
                       preProcess = c('pca'),
                       trControl = trainControl(method = 'cv'))

print(fit.nb.pca.cv)
## with pca, accuracy went down to 0.653 with fL = 0 and usekernel = TRUE

fit.svm.linear.cv <- train(classe ~ ., data = train, method = 'svmLinear',
                           trControl = trainControl(method = 'cv'))

print(fit.svm.linear.cv)
## accuracy 0.787

fit.svm.linear.pca.cv <- train(classe ~ ., data = train, method = 'svmLinear',
                           preProcess = c('pca'),
                           trControl = trainControl(method = 'cv'))

print(fit.svm.linear.pca.cv)
## accuracy 0.571

## since pca doesn't always seem to help, try knn without pca
fit.knn.cv <- train(classe ~ ., data = train, method = 'knn',
                        trControl = trainControl(method = 'cv'))
print(fit.knn.cv)
# without pca, accuracy dropped to 0.931


fit.gbm.cv <- train(classe ~ ., data = train, method = 'gbm',
                    trControl = trainControl(method = 'cv'))
print(fit.gbm.cv)
# accuracy 0.963 for n.trees = 150, interaction.depth = 3,
# shrinkage = 0.1 and n.minobsinnode = 10. 

fit.gbm.pca.cv <- train(classe ~ ., data = train, method = 'gbm',
                        preProcess = c('pca'),
                        trControl = trainControl(method = 'cv'))
print(fit.gbm.pca.cv)
# accuracy 0.825 for n.trees = 150, interaction.depth = 3,
# shrinkage = 0.1 and n.minobsinnode = 10.


# top 5 models
preds.knn <- predict(fit.knn.cv, test)
preds.knn.pca <- predict(fit.knn.pca.cv, test)
preds.rf <- predict(fit.rf.cv, test)
preds.rf.pca <- predict(fit.rf.pca.cv, test)
preds.gbm <- predict(fit.gbm.cv, test)

preds.df <- data.frame(knn = preds.knn,
                       knn.pca = preds.knn.pca,
                       rf = preds.rf,
                       rf.pca = preds.rf.pca,
                       gbm = preds.gbm)

# find most frequent

preds.df$best <- apply(preds.df, 1, function(x) {names(which.max(table(x)))})

preds.df$freq <- apply(preds.df, 1, function(x) {sum(x[1:5] == x[6])/5})


## from the course website, modified based on the note to convert answers
## from factor to charcater
answers = as.character(preds.df$best)
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

setwd("predictions/")
pml_write_files(answers)
