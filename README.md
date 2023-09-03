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
































































































