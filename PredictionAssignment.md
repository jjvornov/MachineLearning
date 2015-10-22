# Prediction Assignment
###October 21, 2015
###James Vornov, MD, PhD

##Load data
The data is in csv format in the working directory and is loaded using read.csv. It took me a while to realize that cleaning is key. Use R NA's and remove the columns that are all NAs in the set.

```r
## Loading the data
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
pml.training<-read.csv("pml-training.csv",na.strings = c("NA", ""))
clean<- pml.training[,-colSums(is.na(pml.training)) == 0]
pml.testing<-read.csv("pml-testing.csv",na.strings = c("NA", ""))
##Note that the data is somewhat messy, so we'll deal only with numeric predictors
inTrain <- createDataPartition(y=clean$classe,
                              p=0.7, list=FALSE)
training <- clean[inTrain,]
testing <- clean[-inTrain,]
##the first 7 columns are subject and timing information
training<-training[,8:dim(training)[2]]
dim(pml.training);dim(clean);dim(training)
```

```
## [1] 19622   160
```

```
## [1] 19622    60
```

```
## [1] 13737    53
```

```r
summary(training$classe)
```

```
##    A    B    C    D    E 
## 3906 2658 2396 2252 2525
```
We still have a large number of variables and one could pick out features based on box plots if the automated model building runs into trouble. Means are often the same but there are outliers that seem to identify the classification.
![](PredictionAssignment_files/figure-html/unnamed-chunk-2-1.png) ![](PredictionAssignment_files/figure-html/unnamed-chunk-2-2.png) 
With the large number of observations and variables for a classification problem with weak features, a bagging or random tree model is said to provide the best results. Random tree models took longer to run on all predictors than the bagging model, so I focused on the "treebag" method. For validation I used k=10 fold repeats, just once since the results were fine. Note that this runs faster on my 4 core MacBook Pro with doMC loaded.


```
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
## Loading required package: ipred
## Loading required package: plyr
## Loading required package: e1071
```
Looking at the most important variables one can see some separation looking at the graphs in retrospect. 

```r
varImp(modelFit)
```

```
## Loading required package: rpart
```

```
## treebag variable importance
## 
##   only 20 most important variables shown (out of 52)
## 
##                   Overall
## roll_belt          100.00
## yaw_belt            80.81
## pitch_belt          75.11
## pitch_forearm       66.81
## magnet_dumbbell_y   60.56
## roll_forearm        59.41
## magnet_dumbbell_z   55.35
## accel_dumbbell_y    54.10
## roll_dumbbell       36.72
## magnet_belt_y       34.86
## accel_belt_z        33.33
## yaw_arm             32.43
## magnet_belt_z       31.94
## magnet_dumbbell_x   29.84
## accel_dumbbell_z    26.74
## magnet_forearm_z    26.43
## gyros_belt_z        25.09
## accel_forearm_x     23.54
## magnet_belt_x       21.41
## total_accel_belt    20.61
```
Next, the model is checked against the validation subset. This is a second cross validation beyond that in the model building by repeated sampling.  It's cleaned to have just the variables in the training set. The prediction looks quite good and it's 98% accurate with an expected out of sample error of 2% (1-Accuracy).

```r
##fix the validation set
cleantesting<-testing[,names(training)]
pred<-predict(modelFit,cleantesting)
table(pred,cleantesting$classe)
```

```
##     
## pred    A    B    C    D    E
##    A 1663   16    3    2    1
##    B    4 1103    8    2    5
##    C    3   14 1005   13    4
##    D    2    6    9  947    2
##    E    2    0    1    0 1070
```

```r
print("misclassification rate");(sum(pred != cleantesting$classe))/length(pred)
```

```
## [1] "misclassification rate"
```

```
## [1] 0.01648258
```

```r
CM<-confusionMatrix(cleantesting$classe,predict(modelFit,cleantesting))
 CM$overall[1]
```

```
##  Accuracy 
## 0.9835174
```
Now the test set is predicted for grading.

```r
##Now the test
answers <- predict(modelFit, pml.testing)
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(answers)
```
