library(plyr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)

#IMPORT FILE:
us <- read.csv(file="/*****/*******/us_events.csv",
               header=TRUE,sep=",", as.is = TRUE)

###FORMAT VARIABLES:
str(us)

us$user_id <- as.factor(us$user_id)
us$browser <- as.factor(us$browser)
us$city <- as.factor(us$city)
us$device <- as.factor(us$device)
us$os <- as.factor(us$os)
us$region <- as.factor(us$region)
us$search_engine <- as.factor(us$search_engine)
us$Pagekey1 <- as.factor(us$Pagekey1)
us$convert_yn <- as.factor(us$convert_yn)
us$page2 <- as.factor(us$page2)
us$day <- as.factor(us$day)

us$search_engine <- as.character(us$search_engine)
us$search_engine[us$search_engine==""] <- "unknown"
us$search_engine <- as.factor(us$search_engine)

#SUBSET DATA:
subsetUS <- subset(us, select = -c(X, user_id, name, time, cs.home.value,
                                   city, current_url, initial_referrer, referrer,
                                   mp_country_code, page, Pagekey2, ud.income,
                                   ud.monthly.debt.repayments, timestamp, date, 
                                   actiontotal, hr))

##fix outcome variable
levels(subsetUS$convert_yn) <- make.names(levels(factor(subsetUS$convert_yn)))
str(subsetUS$convert_yn)

##########MACHINE LEARNING
##dummy coding variables first:
library(caret)
set.seed(1234)
trainIndex <- createDataPartition(subsetUS$convert_yn, p = .8,
                                  list = FALSE, times = 1)
trainIndex %>% head()

training.y <- subsetUS[trainIndex, "convert_yn"]
testing.y <- subsetUS[-trainIndex, "convert_yn"]

training <- subsetUS %>%
  select(-matches("convert_yn")) %>%
  slice(trainIndex)

testing <- subsetUS %>%
  select(-matches("convert_yn")) %>%
  slice(-trainIndex)
str(training)

##Compute dummy variables for factor variables
dummies <- dummyVars(~ ., data = training, sep = "_")
training <- data.frame( predict(dummies, training))
testing <- data.frame( predict(dummies, testing) )

##Dimension reduction via zero-variance
nzv <- nearZeroVar(training)
if(length(nzv) >0) {
  training <- training[, -nzv];
  testing <- testing[, -nzv]
}
dim(testing)
dim(training)

##Correlations
descrCor <- cor(training)
highlyCorDescr <- findCorrelation(descrCor, cutoff = .75)

if(length(highlyCorDescr) >0) {
  training <- training[, -highlyCorDescr];
  testing <- testing[, -highlyCorDescr]
}

##Linear Dependencies
comboInfo <- findLinearCombos(training)
comboInfo

if(length(comboInfo$remove) >0) {
  training <- training[, -comboInfo$remove];
  testing <- testing[, -comboInfo$remove]
}

str(training)
dim(training)
nrow(training)/ncol(training)

training$convert_yn <- training.y
testing$convert_yn <- testing.y

###FIT A GLM AND SCORE THE TEST DATA SET:
#glmModel <- glm(convert_yn~., data=training, family = binomial)
#summary(glmModel)
#calculate r^2
#library(pscl)
#pR2(glmModel)

##predict on test set
#predglm <- predict(glmModel, newdata=testing, type="response")
#library(pROC)
#roc.glmModel <- pROC::roc(testing$convert_yn, predglm)
#auc.glmModel <- pROC::auc(roc.glmModel)

#plot(roc.glmModel)
#glmImp <- varImp(glmModel)
#glmImp
#plot(glmImp,top = 20)

#alternate way using Caret (for CV purposes)
set.seed(1234)
fitControl <- trainControl(method = "cv",
                           number = 10,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

logModel <- train(convert_yn ~ ., data=training, method = "glm", 
                           family='binomial', metric="ROC", trControl = fitControl)
summary(logModel)
pred.logModel <- as.vector(predict(logModel, newdata=testing, 
                                            type="prob")[,"X1"])
roc.logModel <- pROC::roc(testing$convert_yn, pred.logModel)
auc.logModel <- pROC::auc(roc.logModel)

plot(roc.logModel)
logImp <- varImp(logModel)
logImp
plot(logImp,top = 30)

##Confusion matrix, sensitivity, specificity
pr_log <- predict(logModel, newdata = testing)
log_CM <- confusionMatrix(pr_log, testing$convert_yn)
log_CM

#######################OTHER ALGORITHMS:
##training the model:

fitControl <- trainControl(method = "cv",
                           number = 10,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

#1. DECISION TREE
set.seed(1234)
cartModel <- train(convert_yn ~ ., data=training, method = "rpart", 
                   metric="ROC", trControl = fitControl, tuneLength=5)

pred.cartModel <- as.vector(predict(cartModel, newdata=testing, 
                                    type="prob")[,"X1"])
roc.cartModel <- pROC::roc(testing$convert_yn, pred.cartModel)
auc.cartModel <- pROC::auc(roc.cartModel)

crtImp <- varImp(cartModel)
crtImp
plot(crtImp,top = 20)

#2. GLM BOOST
set.seed(1234)
glmBoostModel <- train(convert_yn ~ ., data=training, method = "glmboost", 
                       metric="ROC", trControl = fitControl, tuneLength=5, 
                       center=TRUE, family=Binomial(link = c("logit")))

pred.glmBoostModel <- as.vector(predict(glmBoostModel, newdata=testing, 
                                        type="prob")[,"X1"])
roc.glmBoostModel <- pROC::roc(testing$convert_yn, pred.glmBoostModel)
auc.glmBoostModel <- pROC::auc(roc.glmBoostModel)

bstImp <- varImp(glmBoostModel)
bstImp
plot(bstImp,top = 20)

#3. Random Forest
set.seed(1234)
rfModel <- train(convert_yn ~ ., data=training, method = "rf", metric="ROC", 
                 trControl = fitControl, verboseIter=TRUE, tuneLength=1)

#alternate way using randomforest package

library(randomForest)
rfmodel <- randomForest(convert_yn ~ ., data = training, ntree=100)
pred.rfModel <- as.vector(predict(rfmodel, newdata=testing, 
                                        type="prob")[,"X1"])
roc.rfModel <- pROC::roc(testing$convert_yn, pred.rfModel)
auc.rfModel <- pROC::auc(roc.rfModel)

rfImp <- varImp(rfmodel)
rfImp
plot(rfImp,top = 20)

##Choosing the best model:
test.auc <- data.frame(model=c("Logistic","GLM Boost","Tree", "Random Forest"),
                       auc=c(auc.glmModel, auc.glmBoostModel, auc.cartModel, 
                             auc.rfModel))
test.auc <- test.auc[order(test.auc$auc, decreasing=TRUE),]
test.auc$model <- factor(test.auc$model, levels=test.auc$model)
test.auc

#plot AUCs for models

aucplot<- ggplot(data=test.auc, aes(x=model, y=auc)) +
  geom_bar(fill = "slate blue2", stat="identity") 

##########################################

#RERUN USING OVERSAMPLING TECHNIQUES

library(ROSE)
training_over <- ovun.sample(convert_yn ~ ., data = training, 
                                method = "over",N = 730868)$data
table(training_over$convert_yn)

#1. Logistic regression (caret, cv)
set.seed(1234)
logModel_over <- train(convert_yn ~ ., data=training_over, method = "glm", 
                          family='binomial', metric="ROC", trControl = fitControl)
summary(logModel_over)

pred.logModel_over <- as.vector(predict(logModel_over, newdata=testing, 
                                           type="prob")[,"X1"])
roc.logModel_over <- pROC::roc(testing$convert_yn, pred.logModel_over)
auc.logModel_over <- pROC::auc(roc.logModel_over)
plot(roc.logModel_over)

logImp_over <- varImp(logModel_over)
logImp_over
plot(logImp_over,top = 20)

##Confusion matrix, sensitivity, specificity
pr_logover <- predict(logModel_over, newdata = testing)
logover_CM <- confusionMatrix(pr_logover, testing$convert_yn)
logover_CM

#2. Decision Tree
set.seed(1234)
cartModel_over <- train(convert_yn ~ ., data=training_over, method = "rpart", 
                           metric="ROC", trControl = fitControl, tuneLength=5)

pred.cartModel_over <- as.vector(predict(cartModel_over, newdata=testing, 
                                            type="prob")[,"X1"])
roc.cartModel_over <- pROC::roc(testing$convert_yn, pred.cartModel_over)
auc.cartModel_over <- pROC::auc(roc.cartModel_over)

crtImp_over <- varImp(cartModel_over)
crtImp_over
plot(crtImp_over,top = 20)

#3. GLM BOOST 
set.seed(1234)
glmBoostModel_over <- train(convert_yn ~ ., data=training_over, 
                               method = "glmboost", metric="ROC", trControl = fitControl, 
                               tuneLength=5, center=TRUE, family=Binomial(link = c("logit")))

pred.glmBoostModel_over <- as.vector(predict(glmBoostModel_over, 
                                                newdata=testing, type="prob")[,"X1"])
roc.glmBoostModel_over <- pROC::roc(testing$convert_yn,
                                       pred.glmBoostModel_over)
auc.glmBoostModel_over <- pROC::auc(roc.glmBoostModel_over)

bstImp_over <- varImp(glmBoostModel_over)
bstImp_over
plot(bstImp_over,top = 20)

#4. RANDOM FOREST
set.seed(1234)
library(randomForest)
rfmodel_over <- randomForest(convert_yn ~ ., data = training_over, ntree=100)
pred.rfModel_over <- as.vector(predict(rfmodel_over, newdata=testing, 
                                          type="prob")[,"X1"])
roc.rfModel_over <- pROC::roc(testing$convert_yn, pred.rfModel_over)
auc.rfModel_over <- pROC::auc(roc.rfModel_over)

rfImp_over <- varImp(rfmodel_over)
rfImp_over
plot(rfImp_over,top = 20)

###################RERUN USING UNDERSAMPLING TECHNIQUES (both N = 141)

summary(training$convert_yn)
training_under <- ovun.sample(convert_yn ~ ., data = training, 
                                 method = "under", N = 282, seed = 1)$data
table(training_under$convert_yn)

#1. Logistic regression (caret, cv)
set.seed(1234)
logModel_under <- train(convert_yn ~ ., data=training_under, method = "glm", 
                           family='binomial', metric="ROC", trControl = fitControl)

pred.logModel_under <- as.vector(predict(logModel_under, newdata=testing, 
                                            type="prob")[,"X1"])
roc.logModel_under <- pROC::roc(testing$convert_yn, pred.logModel_under)
auc.logModel_under <- pROC::auc(roc.logModel_under)

plot(roc.logModel_under)
logImp_under <- varImp(logModel_under)
logImp_under
plot(logsubImp_under,top = 20)

#2. Decision Tree
set.seed(1234)
cartModel_under <- train(convert_yn ~ ., data=training_under, method = "rpart", 
                            metric="ROC", trControl = fitControl, tuneLength=5)

pred.cartModel_under <- as.vector(predict(cartModel_under, newdata=testing, 
                                             type="prob")[,"X1"])
roc.cartModel_under <- pROC::roc(testing$convert_yn, pred.cartModel_under)
auc.cartModel_under <- pROC::auc(roc.cartModel_under)

crtImp_under <- varImp(cartModel_under)
crtImp_under
plot(crtsubImp_under,top = 20)

#3. GLM BOOST
set.seed(1234)
glmBoostModel_under <- train(convert_yn ~ ., data=training_under, 
                                method = "glmboost", metric="ROC", trControl = fitControl, 
                                tuneLength=5, center=TRUE, family=Binomial(link = c("logit")))
summary(glmBoostModel_under)
pred.glmBoostModel_under <- as.vector(predict(glmBoostModel_under, 
                                                 newdata=testing, type="prob")[,"X1"])
roc.glmBoostModel_under <- pROC::roc(testing$convert_yn,
                                        pred.glmBoostModel_under)
auc.glmBoostModel_under <- pROC::auc(roc.glmBoostModel_under)

bstImp_under <- varImp(glmBoostModel_under)
bstImp_under
plot(bstImp_under,top = 20)

#4. RANDOM FOREST
#library(randomForest)
#rfsubmodel_under <- randomForest(convert_yn ~ ., data = trainingsub_under, ntree=100)
#pred.rfsubModel_under <- as.vector(predict(rfsubmodel_under, newdata=testing_sub, 
#                                           type="prob")[,"X1"])
#roc.rfsubModel_under <- pROC::roc(testing_sub$convert_yn, pred.rfsubModel_under)
#auc.rfsubModel_under <- pROC::auc(roc.rfsubModel_under)

#rfsubImp_under <- varImp(rfsubmodel_under)
#rfsubImp_under
#plot(rfsubImp_under,top = 20)

#using Caret (CV)
set.seed(1234)
rfModel_under2 <- train(convert_yn ~ ., data=training_under, method = "rf", 
                           metric="ROC", trControl = fitControl, verboseIter=TRUE, 
                           tuneLength=5)
summary(rfModel_under2)
pred.rfModel_under2 <- as.vector(predict(rfModel_under2, newdata=testing, 
                                            type="prob")[,"X1"])

roc.rfModel_under2 <- pROC::roc(testing$convert_yn, pred.rfModel_under2)
auc.rfModel_under2 <- pROC::auc(roc.rfModel_under2)

rfImp_under2 <- varImp(rfModel_under2)
rfImp_under2
plot(rfsubImp_under,top = 20)

##Confusion matrix, sensitivity, specificity
pr_logunder <- predict(rfModel_under2, newdata = testing)
logunder_CM <- confusionMatrix(pr_logunder, testing$convert_yn)
logunder_CM

########################################SMOTE (using the 'us' csv file):
##this is using the ROSE package; using 80/20 data partitioning

training_rose <- ROSE(convert_yn ~ ., data = training, seed = 1)$data
table(training_rose$convert_yn)

set.seed(1234)
logModel_rose <- train(convert_yn ~ ., data=training_rose, method = "glm", 
                        family='binomial', metric="ROC", trControl = fitControl)
summary(logModel_rose)

pred.logModel_rose <- as.vector(predict(logModel_rose, newdata=testing, 
                                         type="prob")[,"X1"])
roc.logModel_rose <- pROC::roc(testing$convert_yn, pred.logModel_rose)
auc.logModel_rose <- pROC::auc(roc.logModel_rose)

plot(roc.logModel_rose)
logImp_rose <- varImp(logModel_rose)
logImp_rose
plot(logImp_rose,top = 20)

#2. Decision Tree
set.seed(1234)
cartModel_rose <- train(convert_yn ~ ., data=training_rose, method = "rpart", 
                         metric="ROC", trControl = fitControl, tuneLength=5)
summary(cartModel_rose)
pred.cartModel_rose <- as.vector(predict(cartModel_rose, newdata=testing, 
                                          type="prob")[,"X1"])
roc.cartModel_rose <- pROC::roc(testing$convert_yn, pred.cartModel_rose)
auc.cartModel_rose <- pROC::auc(roc.cartModel_rose)

crtImp_rose <- varImp(cartModel_rose)
crtImp_rose
plot(crtImp_rose,top = 20)

#3. GLM BOOST
set.seed(1234)
glmBoostModel_rose <- train(convert_yn ~ ., data=training_rose, 
                             method = "glmboost", metric="ROC", trControl = fitControl, 
                             tuneLength=5, center=TRUE, family=Binomial(link = c("logit")))

pred.glmBoostModel_rose <- as.vector(predict(glmBoostModel_rose, 
                                              newdata=testing, type="prob")[,"X1"])
roc.glmBoostModel_rose <- pROC::roc(testing$convert_yn,
                                     pred.glmBoostModel_rose)
auc.glmBoostModel_rose <- pROC::auc(roc.glmBoostModel_rose)

bstImp_rose <- varImp(glmBoostModel_rose)
bstImp_rose
plot(bstImp_rose,top = 20)

#4. RANDOM FOREST
library(randomForest)
set.seed(1234)
rfmodel_rose <- randomForest(convert_yn ~ ., data = training_rose, ntree=100)
pred.rfModel_rose <- as.vector(predict(rfmodel_rose, newdata=testing, 
                                           type="prob")[,"X1"])
roc.rfModel_rose <- pROC::roc(testing$convert_yn, pred.rfModel_rose)
auc.rfModel_rose <- pROC::auc(roc.rfModel_rose)

rfImp_rose <- varImp(rfmodel_rose)
rfImp_rose
plot(rfImp_rose,top = 20)

### CONVERTER VS. NON-CONVERTER GROUP COMPARISONS

##diff between groups on device usage

library(MASS) 
dev = table(subsetUS$convert_yn, subsetUS$device) 
dev
chisq.test(dev)

##diff between groups on OS
subsetUS$os <- trimws(subsetUS$os)
os = table(subsetUS$convert_yn, subsetUS$os)
os
chisq.test(os)

##diff between groups on state
subsetUS$region <- trimws(subsetUS$region)
reg = table(subsetUS$convert_yn, subsetUS$region)
reg
chisq.test(reg)

##diff between groups on search engine
subsetUS$search_engine <- trimws(subsetUS$search_engine)
se = table(subsetUS$convert_yn, subsetUS$search_engine)
se
chisq.test(se)

##diff between groups on Page key
subsetUS$Pagekey1 <- trimws(subsetUS$Pagekey1)
pk = table(subsetUS$convert_yn, subsetUS$Pagekey1)
pk
chisq.test(pk)

##diff between groups on page type
pg <- table(subsetUS$convert_yn, subsetUS$page2)
pg
chisq.test(pg)

##diff between groups on day of the week
dy <- table (subsetUS$convert_yn, subsetUS$day)
dy
chisq.test(dy)

#diff between converters and nonconverters on page actions:

#pgld
t.test(subsetUS$pgld ~ subsetUS$convert_yn)
by(subsetUS$pgld, subsetUS$convert_yn, sd)

#inpsub
t.test(subsetUS$inpsub ~ subsetUS$convert_yn)
by(subsetUS$inpsub, subsetUS$convert_yn, sd)

#oninpclose
t.test(subsetUS$oninpclose ~ subsetUS$convert_yn)
by(subsetUS$oninpclose, subsetUS$convert_yn, sd)

#morttypechange
t.test(subsetUS$morttypechnge ~ subsetUS$convert_yn)
by(subsetUS$morttypechnge, subsetUS$convert_yn, sd)

#mortclick
t.test(subsetUS$mortclck ~ subsetUS$convert_yn)
by(subsetUS$mortclck, subsetUS$convert_yn, sd)

#fltlblmrtclck
t.test(subsetUS$fltlblmrtclck ~ subsetUS$convert_yn)
by(subsetUS$fltlblmrtclck, subsetUS$convert_yn, sd)

#rfshminimrtclck
t.test(subsetUS$rfshminimrtclck ~ subsetUS$convert_yn)
by(subsetUS$rfshminimrtclck, subsetUS$convert_yn, sd)

#clckviewmrtg
t.test(subsetUS$clckviewmrtg ~ subsetUS$convert_yn)
by(subsetUS$clckviewmrtg, subsetUS$convert_yn, sd)

#nxtstpclck
t.test(subsetUS$nxtstpclck ~ subsetUS$convert_yn)
by(subsetUS$nxtstpclck, subsetUS$convert_yn, sd)

#rocket
t.test(subsetUS$rocket ~ subsetUS$convert_yn)
by(subsetUS$rocket, subsetUS$convert_yn, sd)

#veteran
t.test(subsetUS$veteran ~ subsetUS$convert_yn)
by(subsetUS$veteran, subsetUS$convert_yn, sd)

#action total1
t.test(subsetUS$actiontotal1 ~ subsetUS$convert_yn)
by(subsetUS$actiontotal1, subsetUS$convert_yn, sd)

#hour
t.test(subsetUS$dechr ~ subsetUS$convert_yn)
by(subsetUS$dechr, subsetUS$convert_yn, sd)

# Overlaid histograms
a <- ggplot(subsetUS, aes(x=dechr, color=convert_yn)) +
  geom_histogram(fill="white", position="identity")

########################USING SMOTE ALGORITHM (data partition = 0.5)

library(caret)
set.seed(1234)
trainIndex5 <- createDataPartition(subsetUS$convert_yn, p = .5,
                                  list = FALSE, times = 1)
trainIndex5 %>% head()

training5.y <- subsetUS[trainIndex5, "convert_yn"]
testing5.y <- subsetUS[-trainIndex5, "convert_yn"]

training5 <- subsetUS %>%
  select(-matches("convert_yn")) %>%
  slice(trainIndex5)

testing5 <- subsetUS %>%
  select(-matches("convert_yn")) %>%
  slice(-trainIndex5)
str(training5)

##Compute dummy variables for factor variables
dummies <- dummyVars(~ ., data = training5, sep = "_")
training5 <- data.frame( predict(dummies, training5))
testing5 <- data.frame( predict(dummies, testing5) )

##Dimension reduction via zero-variance
nzv <- nearZeroVar(training5)
if(length(nzv) >0) {
  training5 <- training5[, -nzv];
  testing5 <- testing5[, -nzv]
}
dim(testing5)
dim(training5)

##Correlations
descrCor <- cor(training5)
highlyCorDescr <- findCorrelation(descrCor, cutoff = .75)

if(length(highlyCorDescr) >0) {
  training5 <- training5[, -highlyCorDescr];
  testing5 <- testing5[, -highlyCorDescr]
}

##Linear Dependencies
comboInfo <- findLinearCombos(training5)
comboInfo

if(length(comboInfo$remove) >0) {
  training5 <- training5[, -comboInfo$remove];
  testing5 <- testing5[, -comboInfo$remove]
}

str(training5)
dim(training5)
nrow(training5)/ncol(training5)

training5$convert_yn <- training5.y
testing5$convert_yn <- testing5.y

library(DMwR)
#equalize proportions
training5 <- SMOTE(convert_yn ~ ., training5, perc.over = 100, perc.under=200)
training5$convert_yn <- as.numeric(training5$convert_yn)
prop.table(table(training5$convert_yn))

#train using same linear regression
#0 is now 1; 1 is now 2. Recode back

library(car)
training5$convert_yn<-recode(training5$convert_yn,"1=0")
training5$convert_yn<-recode(training5$convert_yn,"2=1")
training5$convert_yn <- as.factor(training5$convert_yn)
str(training5$convert_yn)

levels(training5$convert_yn) <- make.names(levels(factor(training5$convert_yn)))

#glmModel_smote <- glm(convert_yn~ . , data=training5, family=binomial)
#pred.glmModel_smote <- predict(glmModel_smote, newdata=testing5, type="response")
#library(pROC)
#roc.glmModel_smote <- pROC::roc(testing5$convert_yn, pred.glmModel_smote)
#auc.glmModel_smote <- pROC::auc(roc.glmModel_smote)

set.seed(1234)
logModel_smote <- train(convert_yn ~ ., data=training5, method = "glm", 
                       family='binomial', metric="ROC", trControl = fitControl)
summary(logModel_smote)
pred.logModel_smote <- as.vector(predict(logModel_smote, newdata=testing5, 
                                        type="prob")[,"X1"])
roc.logModel_smote <- pROC::roc(testing5$convert_yn, pred.logModel_smote)
auc.logModel_smote <- pROC::auc(roc.logModel_smote)

plot(roc.logModel_smote)
logImp_smote <- varImp(logModel_smote)
logImp_smote
plot(logImp_smote,top = 20)

#confusion matrix
pr_logsmotelog <- predict(logModel_smote, newdata = testing)
logsmotelog_CM <- confusionMatrix(pr_logsmotelog, testing$convert_yn)
logsmotelog_CM

#2. Decision Tree
set.seed(1234)
cartModel_smote <- train(convert_yn ~ ., data=training5, method = "rpart", 
                        metric="ROC", trControl = fitControl, tuneLength=5)

pred.cartModel_smote <- as.vector(predict(cartModel_smote, newdata=testing5, 
                                         type="prob")[,"X1"])
roc.cartModel_smote <- pROC::roc(testing5$convert_yn, pred.cartModel_smote)
auc.cartModel_smote <- pROC::auc(roc.cartModel_smote)

crtImp_smote <- varImp(cartModel_smote)
crtImp_smote
plot(crtImp_smote,top = 20)

#3. GLM BOOST
set.seed(1234)
glmBoostModel_smote <- train(convert_yn ~ ., data=training5, 
                            method = "glmboost", metric="ROC", trControl = fitControl, 
                            tuneLength=5, center=TRUE, family=Binomial(link = c("logit")))
summary(glmBoostModel_smote)

pred.glmBoostModel_smote <- as.vector(predict(glmBoostModel_smote, 
                                             newdata=testing5, type="prob")[,"X1"])
roc.glmBoostModel_smote <- pROC::roc(testing5$convert_yn,
                                    pred.glmBoostModel_smote)
auc.glmBoostModel_smote <- pROC::auc(roc.glmBoostModel_smote)

bstImp_smote <- varImp(glmBoostModel_smote)
bstImp_smote
plot(bstImp_smote,top = 20)

##Confusion matrix, sensitivity, specificity
pr_boostsmote <- predict(glmBoostModel_smote, newdata = testing)
boostsmote_CM <- confusionMatrix(pr_boostsmote, testing$convert_yn)
boostsmote_CM

#4. RANDOM FOREST
set.seed(1234)
rfModel_smote2 <- train(convert_yn ~ ., data=training5, method = "rf", 
                        metric="ROC", trControl = fitControl, verboseIter=TRUE, 
                        tuneLength=5)
summary(rfModel_smote2)
pred.rfModel_smote2 <- as.vector(predict(rfModel_smote2, newdata=testing5, 
                                         type="prob")[,"X1"])

roc.rfModel_smote2 <- pROC::roc(testing5$convert_yn, pred.rfModel_smote2)
auc.rfModel_smote2 <- pROC::auc(roc.rfModel_smote2)

rfImp_smote2 <- varImp(rfModel_smote2)
rfImp_smote2
plot(rfImp_smote2,top = 20)

##Confusion matrix, sensitivity, specificity
pr_rfsmote <- predict(rfModel_smote2, newdata = testing)
rfsmote_CM <- confusionMatrix(pr_rfsmote, testing$convert_yn)
rfsmote_CM

####RUN A PCA on the origina data set to see whether any clustering is 
###occurring. If so, this may further support the RF as the best algorithm

str(training)
cleanjoin <- bind_rows(training, testing)
dim(cleanjoin)
str(cleanjoin)

pca_us <- princomp(cleanjoin[-31])
summary(pca_us)
pca_us$loadings
screeplot(pca_us)
plot(pca_us, type = "l")

library(ggfortify)
cleanjoin_num <- subset(cleanjoin, select=-c(convert_yn, pgld_z, oninp_z, dechr_z))
pcaplot1 <- autoplot(prcomp(cleanjoin_num))
autoplot(prcomp(cleanjoin_num), data = cleanjoin, colour = 'convert_yn')

##redo using all 120 variables?
str(training)
str(testing)
cleanjoin120 <- bind_rows(training, testing)
dim(cleanjoin120)

pca_us120 <- princomp(cleanjoin120)
summary(pca_us120)
pca_us120$loadings
screeplot(pca_us120)
plot(pca_us120, type = "l")

pcaplot2 <- autoplot(prcomp(cleanjoin120))
autoplot(prcomp(cleanjoin120), data = subsetUS, colour = 'convert_yn')
