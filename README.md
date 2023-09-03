# PROJECT-2 RETAIL

## PROJECT DESCRIPTION
Here in this project we are doing analysis on Real estate Price prediction using the Logistic Regression, Random Forest, Decision Tree, xgboost and GBM model and choose the best model and Tune the model with Hyper Parametre tuning then prediction for Whether we can open the store or not. Also we are going to explain the entire project on the basis of Data Analysis procedure of Ask, Prepare, Process, Analyze, Share and Report using R-Programming.

## PROBLEM STATEMENT: ASK 
A retail giant needs to plan its new store openings over the next 12-18 months and
has multiple locations where the stores can be opened. It wants to understand which locations would be
the best to open new stores in terms of market size and potential revenues. Relevant data regarding
locations, population, sales and revenues of key products for stores opened in the past was provided to
be analyzed and for model building.

## COLLECT AND IMPORTING THE DATASET: PREPARE
Once i have collected the data from stakeholder i have received the Dataset which has 3338 observations and 17 variables. 
In the initial step im importing the  Dataset to the R and started to work on the Dataset.

```
getwd()
setwd("C:/DATA ANALYTICS/R PROGRAMMING/PROJECT 2") #Setting the working directory
p2_train = read.csv("store_train.csv",stringsAsFactors = FALSE) #Importing the Train dataset
```

## DATA CLEAINING PROCEDURE : PROCESS
After importing the Dataset we should understand the Dataset so that we can identify that the  NA values, Dummy creation, Unnecessary Variables, Converting variables to numeric / Factor/ Integer and other errors in the dataset.

```view(head(p2_train))``` #Train dataset

![image](https://github.com/swasthik62/project2/assets/125183564/d864fdd2-20c4-480e-9e55-a1362b7d3190)

### Deal with NA values

After importing the dataset we need to understand the Missing values and NA's 

```colSums(is.na(p2_train))``` #We have get the data of NA values accumulated in Train data

![image](https://github.com/swasthik62/project2/assets/125183564/0397fc52-80f7-493a-9aba-4023dfe4419f)

We can also plot the Tables of NAs using the ```vis_dat``` function.

```
library(visdat) #importing the librariries
vis_dat(p2_train) # to plot the train dataset
```
![image](https://github.com/swasthik62/project2/assets/125183564/e37c5a03-3b05-42f5-9c63-8b664bf23f6a)

There are many missing values and we need to convert some Variables into Numerical values, And We need to create Dummies for certain Variables.

As per the Variable Analysis i have found some inputs and applied it on the dataset.

```
Id :  drop the variable as this has no correlation.
sales0 : To numeric
sales1: To numeric
sales2 : To numeric
sales3 : To numeric
sales4 : To numeric
country : To numeric
State : To numeric
CouSub : To numeric
countyname : club low frequency cateorories and then create dummies then conver to numeric.
storecode : Converting into dummies ::not be used  but can be source of a feature
Areaname :  drop the variable as this has no correlation.
countytownname : drop the variable as this has no correlation.
population : To numeric
state_alpha : club low frequency cateorories and then create dummies then conver to numeric.
store_Type : club low frequency cateorories and then create dummies then conver to numeric.
store : categorical 1/0 : target indicator var 1=opened 0=not opened 
```

### Data Cleaning Process: 

Im doing some data cleansing procedure using recipe functions

Aim :
1. To remove the unwanted variables
2. Convert the Variables into Numeric/Charecter/Factor.
3. To create dummies
4. To handle the missing values followed by Mean, Median, mode etc.

#converting target var as factor
```
p2_train$store=as.factor(p2_train$store)
```

```
library(tidymodels)
dp_pipe=recipe(store~.,data=p2_train) %>%
  update_role(Id,countytownname,
              Areaname,new_role = "drop_vars") %>%
  update_role(State,CouSub,country,population,sales1,sales0,
              sales2,sales3,sales4,new_role ="to_numeric") %>% 
  update_role(state_alpha,store_Type,countyname,storecode,new_role = "to_dummies") %>%
  step_rm(has_role("drop_vars")) %>%
  step_unknown(has_role("to_dummies"),new_level = "--missing--") %>%
  step_other(has_role("to_dummies"),threshold = 0.02,other = "--other--") %>%
  step_dummy(has_role("to_dummies")) %>%
  step_mutate_at(has_role("to_numeric"),fn=as.numeric) %>% 
  step_impute_median(all_numeric(),-all_outcomes())
```
Once the ```dp_pipe``` is created we need to ```prep``` and ```bake``` the dp_pipe in the next step

```
dp_pipe=prep(dp_pipe)
train=bake(dp_pipe,new_data = NULL)
```

After the recipe funtion there will be no Na or missing values and also there will additional variables were added due to the Create Dummy funtion. Lets check the dataset.

again im running the is.na querry to check the NA's

![image](https://github.com/swasthik62/project2/assets/125183564/cb2d0223-c58c-4fb8-8aa3-da1e830ee041)

``` vis_dat(train) # to plot the train dataset ``` 

![image](https://github.com/swasthik62/project2/assets/125183564/129b379b-fc53-40ce-aeaf-bdd160b184c4)

We can see there is no missing values, all the Variables are converted into  numerical values and Dummies are created.

### Splitting the Dataset
Split the `train` dataset into two `trn` and `tst` to check the model performance.

```
set.seed(2) #using this function system cannot ablter any rows throughout the process.
s=sample(1:nrow(hs_train),0.8*nrow(hs_train)) #we are spliting the data into 80% and 20%
trn=hs_train[s,]
tst=hs_train[-s,]
```

One the Dataset has been splitted we can check the performance of dataset using Different Models.

### Implement and Check with the different models

We are imposing various models to check the performance of the Dataset.

#### Logistic Regression Model:
Here we are using Linear Model to remove the VIF values where variables has greater than 10 and this model nothing to do with the classification problem.

```
for_vif=lm(store~.-store_Type_X..other..
           -state_alpha_X..other..
           -sales3-sales0-sales2,data = trn)

sort(vif(for_vif),decreasing = T)[1:3]

summary(for_vif)
```
Now we can observe all the VIF values are below 10 and now we can proceed with the Logistic regresson.

```
log_fit = glm(store~.-store_Type_X..other..
          -state_alpha_X..other..
          -sales3-sales0-sales2,data = trn)
summary(log_fit)
```
We can run the AUC values and check the performance on thisdataset

### Calculating the AUC score
```
val.score=predict(log_fit,newdata = trn, type = "response")
pROC::auc(pROC::roc(trn$store,val.score)) #0.753

val.score1=predict(log_fit,newdata = tst, type = "response")
pROC::auc(pROC::roc(tst$store,val.score1)) ## 0.7252
```

We can see there are many non colinear variables we need to remove them based on the P-Value

```
fit=stats::step(log_fit) ##which will drop the high p value

formula(log_fit) # Which will f=give the list of variables
log_fit.final=glm(store~ sales1 + sales2 +  sales4 +  
              State + population +  storecode_X..other.. + 
              state_alpha_GA +   
              state_alpha_MA +   
              state_alpha_NH + state_alpha_TX + state_alpha_VA + 
              state_alpha_VT,data=trn)
summary(log_fit.final)
```
once we are done with this we can proceed with the AUC score

### Calculating the AUC score
```
val.score=predict(log_fit.final,newdata = trn, type = "response")
pROC::auc(pROC::roc(trn$store,val.score)) #0.7598

val.score1=predict(log_fit.final,newdata = tst, type = "response")
pROC::auc(pROC::roc(tst$store,val.score1)) #0.7399
```
There is no significant differences and we are moving further with the other models

### Decison Tree Model
Now we are implementing the decision tree model on the test and tran dataset and check the model performance on the dataset.

```
dtree= tree(store~.,data=trn)

val.score=predict(dtree,newdata = trn,)
pROC::auc(pROC::roc(trn$store,val.score)) #0.7492

val.score1=predict(dtree,newdata = tst)
pROC::auc(pROC::roc(tst$store,val.score1)) #0.7251
```
We can observe that there is not much significance difference in the Dataset as the AUC score lies in the 0.7 Range, so inorder to bring the values up we should go for other models.

### Random Forest Model
Now we are implementing the Random Forest model on the test and tran dataset and check the model performance on the dataset.

```
rrf = randomForest(store~.,data=trn)
val.score=predict(rrf,newdata = trn, type="response")
pROC::auc(pROC::roc(trn$store,val.score)) #0.9976

val.score1=predict(rrf,newdata = tst) 
pROC::auc(pROC::roc(tst$store,val.score1)) #0.8106
```
We can see there is a good values when we use the RF model and we got the AUC  score in ```tst``` dataset as 0.997 so in further we are doing Random Forest Hyper Parameter Tuning using Gradient Boosting Machine.


### Random Forest Hyper Parameter Tuning
Hyperparameter tuning allows us to tweak model performance for optimal results.

**Parameter Grid Definition (param):**
```
param = list(
  interaction.depth = c(1:7),
  n.trees = c(50, 100, 200, 500, 700),
  shrinkage = c(0.1, 0.01, 0.001),
  n.minobsinnode = c(1, 2, 5, 10)
)
```
This section defines a list named param that contains sets of hyperparameters for the GBM model that we want to tune. It's a grid of hyperparameters with various possible values. For example:

interaction.depth: Varies from 1 to 7.
n.trees: Takes on values 50, 100, 200, 500, and 700.
shrinkage: Includes values 0.1, 0.01, and 0.001.
n.minobsinnode: Contains values 1, 2, 5, and 10.
These hyperparameters are essential configuration settings for the Gradient Boosting Machine model.

**Custom Function for Subsetting Parameters (subset_paras):**
```
subset_paras = function(full_list_para, n = 10) {
  all_comb = expand.grid(full_list_para)
  s = sample(1:nrow(all_comb), n)
  subset_para = all_comb[s, ]
  return(subset_para)
}
```
This custom function, subset_paras, takes in a list of hyperparameters (full_list_para) and a number n. It does the following:

Generates all possible combinations of hyperparameters using expand.grid.
Randomly selects n combinations from all the possibilities.
Returns a subset of n hyperparameter combinations.
This function is used to sample different sets of hyperparameters for each iteration of the hyperparameter tuning process.

**Number of Trials (num_trials):**
```
num_trials = 10
```
num_trials specifies how many iterations of hyperparameter tuning you want to perform. In this case, it's set to 10, meaning that the script will explore 10 different sets of hyperparameters.

**Custom Cost Function (mycost_auc):**
```
mycost_auc = function(y, yhat) {
  roccurve = pROC::roc(y, yhat)
  score = pROC::auc(roccurve)
  return(score)
}
```
mycost_auc is a custom cost function that calculates the area under the Receiver Operating Characteristic (ROC) curve (AUC) as the model's performance metric. It takes two arguments:

y: The true labels.
yhat: The predicted probabilities.
This function computes the AUC score using the pROC package and returns it as the performance metric for hyperparameter tuning.

**GBM Model Tuning Loop:**
```
myauc = 0
library(gbm)
for (i in 1:num_trials) {
  print(paste('starting iteration :', i))
  params = my_params[i, ]
  
  k = cvTuning(
    gbm,
    store ~ . - store_Type_X..other.. - state_alpha_X..other.. - sales3 - sales0 - sales2,
    data = train,
    tuning = params,
    args = list(distribution = "bernoulli"),
    folds = cvFolds(nrow(train), K = 10, type = "random"),
    cost = mycost_auc,
    seed = 2,
    predictArgs = list(type = "response", n.trees = params$n.trees)
  )
  score.this = k$cv[, 2]
  
  if (score.this > myauc) {
    print(params)
    myauc = score.this
    print(myauc)
    best_params = params
  }
  
  print('DONE')
}
```
This section of the script is where the actual hyperparameter tuning occurs:

It uses a for loop to iterate through num_trials (in this case, 10 iterations).
For each iteration:
It selects a set of hyperparameters (params) from my_params, which were sampled earlier.
Calls cvTuning to perform k-fold cross-validation on the GBM model.
The target variable is store, and certain variables (store_Type_X..other.., state_alpha_X..other.., sales3, sales0, sales2) are excluded from the predictor variables.
The specified hyperparameters are passed as tuning parameters, and the AUC is used as the cost function.
The results of cross-validation are stored in k.
It compares the AUC score (score.this) from the current iteration with the best AUC score seen so far (myauc). If the current score is better, it updates myauc and records the best hyperparameters in best_params.

**Setting Best Hyperparameters:**
```
best_params = data.frame(
  interaction.depth = 6,
  n.trees = 700,
  shrinkage = 0.01,
  n.minobsinnode = 1
)
```
In this part of the script, we explicitly set the best hyperparameters (best_params) to specific values.

**Model Building with Best Hyperparameters:**

```
myauc
train$store = as.numeric(train$store)

rg.gbm.final = randomForest(
  store ~ . - store_Type_X..other.. - state_alpha_X..other.. - sales3 - sales0 - sales2,
  data = train,
  n.trees = best_params$n.trees,
  n.minobsinnode = best_params$n.minobsinnode,
  shrinkage = best_params$shrinkage,
  interaction.depth = best_params$interaction.depth,
  distribution = "bernoulli"
)
```

This section performs the following tasks:
myauc is printed, displaying the AUC score obtained using the best hyperparameters from the tuning process.
train$store is converted to numeric. This is often done to ensure the target variable is in the correct format for modeling.
A Random Forest model (rg.gbm.final) is constructed using the best hyperparameters (best_params) obtained from the tuning process. The model is built using the randomForest function.
The formula store ~ . - store_Type_X..other.. - state_alpha_X..other.. - sales3 - sales0 - sales2 defines the model formula, specifying the target variable (store) and predictor variables. Some variables are excluded from the predictor set.
data = train specifies the training data.
The hyperparameters (n.trees, n.minobsinnode, shrinkage, interaction.depth) are set to the values from best_params.
distribution = "bernoulli" indicates that the target variable follows a Bernoulli distribution, which is typical for classification problems.

### Analysing the Dataset**Making Predictions and Calculating AUC:** 
```
test.score = predict(rg.gbm.final, newdata = train, type = 'response', n.trees = best_params$n.trees)
pROC::auc(pROC::roc(train$store, test.score)) 0.99
```
Finally after the parameter tuning our Train Dataset is performing well and gives the AUC score as 0.997.
and we need to convey our findings to the stakeholders.

Now let`s see the variable impportance of the dataset
``` varImpPlot(rg.gbm.final)```

![image](https://github.com/swasthik62/project2/assets/125183564/9b4fbe93-7c1e-40aa-a11e-35327a54a821)

### Share and Report
As per the Data analysis we can see that thePopulation, sales1, Sales4, Country, State has a very good corelation betwee the Dependent variable, and also we can see the initial AUC score is 0.75 and after tuning we got the AUC as 0.99 and based on this observation we can see that Random Forest worked out to be a better model and Approximately 25
locations were selected to open new stores based on this analysis.
























