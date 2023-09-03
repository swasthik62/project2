# PROJECT-1 RETAIL

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


























































































