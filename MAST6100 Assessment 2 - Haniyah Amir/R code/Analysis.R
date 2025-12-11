# -----------------------------------------------------------------------------
# MAST6100 Classification Project: Drug Consumption Prediction
# -----------------------------------------------------------------------------

library(readr)
library(dplyr)
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data"
drug <- read_csv(url,col_names = FALSE)
colnames(drug) <- c("ID","Age","Gender","Education","Country","Ethnicity","Nscore","Escore","Oscore","Ascore","Cscore","Impulsive","SS","Alcohol","Amphet","Amyl","Benzos","Caff","Cannabis","Choc","Coke","Crack","Ecstasy","Heroin","Ketamine","Legalh","LSD","Meth","Mushrooms","Nicotine","Semer","VSA") #Assigning column names
drug <- drug %>% select (-ID) #Remove ID as it is irrelevant

#The dataset uses 7 categories - ("CL0"= Never used ... "CL6"= Used in the last day)
#Convert the above categories into two binary variables:
  #0 = Non-user(CL0-CL2 -> Not used in the past year)
  #1 = User (CL3-CL6 -> Used in the last year)
drug$Cannabis_bin <- ifelse(drug$Cannabis %in% c("CL0","CL1","CL2"), 0, 1)
drug$Cannabis_bin <- factor(drug$Cannabis_bin)
#Train/Test split
library(caret)

set.seed(123)
train_index <- createDataPartition(drug$Cannabis_bin, p = 0.8, list = FALSE)
train <- drug[train_index, ]
test  <- drug[-train_index, ]
# -----------------------------------------------------------------------------
#Exploratory data analysis
# -----------------------------------------------------------------------------
#1. Class balance
print(table(train$Cannabis_bin))
print(prop.table(table(train$Cannabis_bin)))

#2. Pairwise correlation
library(corrplot)

corrplot(cor(train %>% select(Nscore:SS)), method = "color")

#3. Simple visualisation
library(ggplot2)

ggplot(train, aes(x = Nscore, fill = Cannabis_bin)) +geom_density(alpha = 0.5) + labs(title="Neuroticism by Cannabis Use")
# -----------------------------------------------------------------------------
#Model fitting
# -----------------------------------------------------------------------------
#1.GLM - Logistic Regression
glm_model <- glm(Cannabis_bin ~ Age + Gender + Education + Ethnicity + Nscore + Escore + Oscore + Ascore + Cscore +Impulsive + SS, data = train, family = binomial())

summary(glm_model)

#a.Predictions and AUC
library(pROC)

glm_prob <- predict(glm_model, test, type="response")
glm_roc  <- roc(test$Cannabis_bin, glm_prob)
auc(glm_roc)

#b.ROC plot
plot(glm_roc, col="blue", main="ROC Curve - GLM")

#2.Random forest
library(randomForest)

rf_model <- randomForest(Cannabis_bin ~ . -Cannabis, data=train, ntree = 500, mtry = 4)

#a.Prediction
rf_prob <- predict(rf_model, test, type="prob")[,2]
rf_roc <- roc(test$Cannabis_bin, rf_prob)
auc(rf_roc)

#b.Feature importance
varImpPlot(rf_model)

#3.XGBoost - covert to numeric matrix
library(xgboost)

train_matrix <- model.matrix(Cannabis_bin ~ . -Cannabis, train)[, -1]
test_matrix  <- model.matrix(Cannabis_bin ~ . -Cannabis, test)[, -1]

dtrain <- xgb.DMatrix(train_matrix, label = as.numeric(train$Cannabis_bin) - 1)
dtest  <- xgb.DMatrix(test_matrix, label = as.numeric(test$Cannabis_bin) - 1)

#a.Train
params <- list(objective = "binary:logistic",eval_metric = "auc")

xgb_model <- xgb.train(params = params,data = dtrain,nrounds = 200,watchlist = list(test = dtest),verbose = 0)

#b.AUC
drug_clean <- drug %>% select(-Cannabis)
full_matrix <- model.matrix(Cannabis_bin ~ ., drug_clean)[, -1]
labels <- as.numeric(drug$Cannabis_bin) - 1
set.seed(123)
train_index <- createDataPartition(labels, p=0.8, list=FALSE)

x_train <- full_matrix[train_index, ]
x_test  <- full_matrix[-train_index, ]

y_train <- labels[train_index]
y_test  <- labels[-train_index]
dtrain <- xgb.DMatrix(x_train, label=y_train)
dtest  <- xgb.DMatrix(x_test, label=y_test)

params <- list(objective="binary:logistic", eval_metric="auc")

xgb_model <- xgb.train(params=params, data=dtrain, nrounds=200)
xgb_prob <- predict(xgb_model, x_test)

xgb_roc <- roc(y_test, xgb_prob)
auc(xgb_roc)

#4.Neural Network (keras)
library(keras)

x_train <- as.matrix(train_matrix)
y_train <- as.numeric(train$Cannabis_bin) - 1

x_test <- as.matrix(test_matrix)
y_test <- as.numeric(test$Cannabis_bin) - 1

model <- keras_model_sequential() %>%
  layer_dense(units=32, activation='relu', input_shape=ncol(x_train)) %>%
  layer_dropout(0.2) %>%
  layer_dense(units=16, activation='relu') %>%
  layer_dense(units=1, activation='sigmoid')

model %>% compile(optimizer='adam',loss='binary_crossentropy',metrics=c('accuracy'))

history <- model %>% fit(x_train, y_train,epochs=40, batch_size=32,validation_split=0.2)

nn_prob <- model %>% predict(x_test)
nn_roc <- roc(y_test, nn_prob)
auc(nn_roc)
# -----------------------------------------------------------------------------
#Compare models
# -----------------------------------------------------------------------------
results <- data.frame(Model = c("GLM", "Random Forest", "XGBoost", "Neural Network"),AUC = c(auc(glm_roc), auc(rf_roc), auc(xgb_roc), auc(nn_roc)))

print(results)